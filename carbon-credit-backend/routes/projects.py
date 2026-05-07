# routes/projects.py
# Handles all project lifecycle endpoints:
# submit → verify → status → mint → report

from __future__ import annotations

import datetime as dt
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware

from database import get_db
from models import CreditLedger, FraudFlag, Project, Verification
from pipeline import run_full_pipeline
from schemas import ProjectResponse, ProjectStatus, ProjectSubmit, VerificationResponse

router = APIRouter(prefix="/projects", tags=["projects"])

# ── Blockchain setup — loaded once at module startup ──────────────────────────
# Keep this configurable: the deployed contract may change between testnets / redeploys.
CONTRACT_ADDRESS = os.getenv(
    "CONTRACT_ADDRESS",
    "0x8d0B4Dbd29ae0a52C1B3d2B4568DFE6aF1032285",
)
SEPOLIA_RPC_URL = os.getenv("SEPOLIA_RPC_URL")
PRIVATE_KEY = os.getenv("BLOCKCHAIN_PRIVATE_KEY")
SEPOLIA_EXPLORER_BASE = "https://sepolia.etherscan.io/tx/"

# ABI lives in the root of carbon-credit-backend/
_ABI_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "CarbonCredit.json")
)
with open(_ABI_PATH, "r", encoding="utf-8") as _f:
    CONTRACT_ABI = json.load(_f)["abi"]


def _require_env(name: str, value: Optional[str]) -> str:
    if not value:
        raise HTTPException(status_code=500, detail=f"{name} is not set in .env")
    return value


def _get_w3() -> Web3:
    rpc_url = _require_env("SEPOLIA_RPC_URL", SEPOLIA_RPC_URL)
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
    if not w3.is_connected():
        raise HTTPException(status_code=502, detail="Cannot connect to Sepolia RPC")
    return w3


def _get_contract(w3: Web3):
    return w3.eth.contract(
        address=Web3.to_checksum_address(CONTRACT_ADDRESS),
        abi=CONTRACT_ABI,
    )


def _safe_jsonish(value: Any) -> Any:
    """
    Keep JSON values stable if the DB stores strings, dicts, lists, or JSON-ish text.
    """
    if value is None:
        return None

    if isinstance(value, (dict, list, tuple, int, float, bool)):
        return value

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return value
        try:
            return json.loads(stripped)
        except Exception:
            return value

    return value


def _flatten_image_hashes(image_hashes: Any) -> List[str]:
    """
    Convert DB/image hash payloads into a simple list of strings.
    Supports:
    - list[str]
    - list[dict]
    - dict
    - JSON string
    """
    value = _safe_jsonish(image_hashes)
    flattened: List[str] = []

    if value is None:
        return flattened

    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                candidate = item.get("image_hash") or item.get("hash") or item.get("value")
                if candidate:
                    flattened.append(str(candidate))
            elif item is not None:
                flattened.append(str(item))
        return flattened

    if isinstance(value, dict):
        # Prefer explicit keys first, then values.
        for key in ("image_hash", "hash", "value"):
            candidate = value.get(key)
            if candidate:
                flattened.append(str(candidate))
                return flattened

        for item in value.values():
            if isinstance(item, dict):
                candidate = item.get("image_hash") or item.get("hash") or item.get("value")
                if candidate:
                    flattened.append(str(candidate))
            elif item is not None:
                flattened.append(str(item))
        return flattened

    return [str(value)]


def _collect_locked_checkpoint_hashes(verifications) -> List[Dict[str, Any]]:
    """
    Preserve checkpoint dictionaries exactly as they were stored.

    Why:
    - ndvi_pipeline.py expects locked_hashes to be a list of dicts with a
      'checkpoint' key.
    - Flattening them into strings breaks reproducibility and causes the
      pipeline to skip the locked values.
    """
    locked_hashes: List[Dict[str, Any]] = []

    for v in verifications:
        payload = _safe_jsonish(v.image_hashes)

        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict) and item.get("checkpoint"):
                    locked_hashes.append(item)

        elif isinstance(payload, dict) and payload.get("checkpoint"):
            locked_hashes.append(payload)

        elif payload is not None:
            print(f"[WARNING] Skipping legacy locked hash format in DB: {payload}")

    return locked_hashes


def _extract_primary_image_hash(image_hashes: Any) -> str:
    flattened = _flatten_image_hashes(image_hashes)
    return flattened[0] if flattened else "no_hash"


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _build_verification_key(project_id: str, year_number: int, verification_id: str) -> str:
    """
    Composite ID for on-chain uniqueness.
    This prevents a one-land-one-verification-ever bug and allows yearly re-surveys.
    """
    return f"{project_id}:year:{year_number}:verification:{verification_id}"


def _candidate_chain_keys(project_id: str, year_number: int, verification_id: str) -> List[str]:
    """
    Try the composite key first, but keep legacy candidates so older records
    can still be detected if the contract already stored the plain DB ID.
    """
    composite = _build_verification_key(project_id, year_number, verification_id)
    return [
        composite,
        verification_id,              # legacy / fallback
        f"{project_id}:year:{year_number}",
        project_id,                   # last-resort fallback for older deployments
    ]


def _try_onchain_duplicate_check(contract, project_id: str, year_number: int, verification_id: str) -> Optional[str]:
    """
    Best-effort check against common getter names / mapping getters.
    Returns the first key that appears to be already recorded, otherwise None.

    This is intentionally defensive because the exact Solidity getter name
    may differ between versions of the contract.
    """
    candidate_keys = _candidate_chain_keys(project_id, year_number, verification_id)

    getter_names = [
        "isVerificationRecorded",
        "verificationRecorded",
        "verificationExists",
        "alreadyRecorded",
        "hasVerification",
        "isRecorded",
        "minted",
    ]

    # Try both a one-key mapping getter and a project+key style getter.
    arg_patterns: Sequence[Tuple[Any, ...]] = (
        ("{key}",),
        ("{project_id}", "{key}"),
        ("{project_id}", "{year_number}", "{key}"),
    )

    for getter_name in getter_names:
        fn = getattr(contract.functions, getter_name, None)
        if fn is None:
            continue

        for key in candidate_keys:
            for pattern in arg_patterns:
                try:
                    args: List[str] = []
                    for token in pattern:
                        if token == "{key}":
                            args.append(key)
                        elif token == "{project_id}":
                            args.append(project_id)
                        elif token == "{year_number}":
                            args.append(str(year_number))
                    result = fn(*args).call()
                    if isinstance(result, bool) and result:
                        return key
                    if isinstance(result, int) and result != 0:
                        return key
                except Exception:
                    continue

    return None


def _build_fee_params(w3: Web3) -> Dict[str, int]:
    """
    Prefer EIP-1559 fields, but fall back to legacy gasPrice if the RPC refuses fee_history.
    """
    try:
        fee_history = w3.eth.fee_history(1, "latest", [50])
        base_fee = fee_history["baseFeePerGas"][-1]
        priority = w3.to_wei(2, "gwei")
        max_fee = int(base_fee * 2 + priority)
        return {
            "maxFeePerGas": max_fee,
            "maxPriorityFeePerGas": int(priority),
        }
    except Exception:
        gas_price = int(w3.eth.gas_price)
        return {"gasPrice": int(gas_price * 12 // 10)}  # 20% safety buffer


def _verification_already_persisted(db: Session, verification: Verification) -> bool:
    """
    Local idempotency guard.

    We treat an entry as already persisted if:
    - tx_hash exists, or
    - minted_at exists (when the ORM model maps it), or
    - a ledger row already exists for the verification.
    """
    if getattr(verification, "tx_hash", None):
        return True

    if getattr(verification, "minted_at", None):
        return True

    ledger_exists = (
        db.query(CreditLedger)
        .filter(CreditLedger.verification_id == verification.id)
        .first()
        is not None
    )
    return ledger_exists


def _verification_response_message(is_flagged: bool, decision: str) -> str:
    if decision == "PASS" and not is_flagged:
        return "PASS — Credits minted to wallet."
    if decision == "PASS" and is_flagged:
        return "PASS — Flagged for review. Credits held until cleared."
    return "FAIL — No credits issued."


# ── POST /projects/submit ──────────────────────────────────────────────────────
@router.post("/submit", response_model=ProjectResponse)
def submit_project(payload: ProjectSubmit, db: Session = Depends(get_db)):
    """
    Accept a land claim from a carbon project developer.
    Stores parcel geometry, area, plantation date, and wallet address.
    Returns a project_id for all subsequent calls.
    """
    if not payload.company_wallet:
        raise HTTPException(status_code=400, detail="company_wallet is required")

    try:
        company_wallet = Web3.to_checksum_address(payload.company_wallet)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid company wallet address")

    project = Project(
        company_name=payload.company_name,
        company_wallet=company_wallet,
        coordinates=payload.coordinates,
        area_hectares=payload.area_hectares,
        plantation_date=payload.plantation_date,
        status="pending",
    )

    db.add(project)
    db.commit()
    db.refresh(project)

    return ProjectResponse(
        project_id=str(project.id),
        company_name=project.company_name,
        status="pending",
        message="Project submitted. Call /verify to run satellite analysis.",
    )


# ── POST /projects/{id}/verify ─────────────────────────────────────────────────
@router.post("/{project_id}/verify", response_model=VerificationResponse)
def verify_project(project_id: str, db: Session = Depends(get_db)):
    """
    Runs the full satellite + ML pipeline on the submitted land claim.
    Stores verification result including confidence score, credits, image hashes.
    Flags for review if score is borderline (below pass_threshold + 5).
    """
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if project.status in {"verifying", "minting"}:
        raise HTTPException(status_code=409, detail=f"Project is already {project.status}")

    project.status = "verifying"
    db.commit()

    year_number = (
        db.query(Verification)
        .filter(Verification.project_id == project_id)
        .count()
        + 1
    )

    all_verifications = (
        db.query(Verification)
        .filter(Verification.project_id == project_id)
        .all()
    )

    locked_hashes = _collect_locked_checkpoint_hashes(all_verifications)

    result = run_full_pipeline(
        coordinates=project.coordinates,
        area_hectares=project.area_hectares,
        plantation_date=project.plantation_date,
        locked_hashes=locked_hashes if locked_hashes else None,
    )

    if "error" in result:
        project.status = "failed"
        db.commit()
        raise HTTPException(status_code=500, detail=result["error"])

    verification = Verification(
        project_id=project.id,
        year_number=year_number,
        ndvi_baseline=_as_float(result.get("ndvi_baseline")),
        ndvi_current=_as_float(result.get("dry_season_ndvi")),
        stage1_credits=_as_int(result.get("stage1_credits")),
        tree_cover_pct=_as_float(result.get("current_tree_pct")),
        confidence_score=_as_float(result.get("confidence_score")),
        decision=str(result.get("decision", "FAIL")),
        adjusted_credits=_as_int(result.get("adjusted_credits")),
        buffer_credits=_as_int(result.get("buffer_credits")),
        active_credits=_as_int(result.get("active_credits")),
        image_hashes=_safe_jsonish(result.get("image_hashes")),
    )
    db.add(verification)

    pass_threshold = _as_float(result.get("pass_threshold"))

    # ------------------------------------------------------------------
    # FLAGGING LOGIC
    # ------------------------------------------------------------------
    # PASS threshold:
    #   Below this = FAIL
    #
    # FLAG threshold:
    #   Slightly above pass threshold = suspicious but acceptable
    #
    # Example:
    #   <45       → FAIL
    #   45–50     → PASS + FLAGGED
    #   >50       → CLEAN PASS
    # ------------------------------------------------------------------

    flag_threshold = pass_threshold + 5

    confidence_score = _as_float(result.get("confidence_score"))

    is_flagged = False

    # Only PASS projects can be flagged
    if (
        str(result.get("decision", "FAIL")) == "PASS"
        and confidence_score < flag_threshold
    ):
        is_flagged = True

        flag = FraudFlag(
            project_id=project.id,
            flag_type="low_confidence",
            details={
                "confidence_score": confidence_score,
                "pass_threshold": pass_threshold,
                "flag_threshold": flag_threshold,
                "margin": round(confidence_score - pass_threshold, 2),
                "decision": str(result.get("decision", "FAIL")),
                "year_number": year_number,
                "review_status": "pending_admin_review",
            },
        )

        db.add(flag)

    if str(result.get("decision", "FAIL")) == "PASS":
        ledger = CreditLedger(
            project_id=project.id,
            verification_id=verification.id,
            credits_issued=_as_int(result.get("adjusted_credits")),
            credits_buffer=_as_int(result.get("buffer_credits")),
            credits_active=_as_int(result.get("active_credits")) if not is_flagged else 0,
        )
        db.add(ledger)
        project.status = "passed" if not is_flagged else "flagged_review"
    else:
        project.status = "failed"

    db.commit()
    db.refresh(verification)

    return VerificationResponse(
        project_id=str(project.id),
        year_number=year_number,
        decision=str(result.get("decision", "FAIL")),
        confidence_score=_as_float(result.get("confidence_score")),
        adjusted_credits=_as_int(result.get("adjusted_credits")) if not is_flagged else 0,
        active_credits=_as_int(result.get("active_credits")) if not is_flagged else 0,
        buffer_credits=_as_int(result.get("buffer_credits")),
        tree_cover_pct=_as_float(result.get("current_tree_pct")),
        ndvi_baseline=_as_float(result.get("ndvi_baseline")),
        ndvi_current=_as_float(result.get("dry_season_ndvi")),
        image_hashes=result.get("image_hashes"),
        message=_verification_response_message(is_flagged, str(result.get("decision", "FAIL"))),
    )


# ── GET /projects/{id}/status ──────────────────────────────────────────────────
@router.get("/{project_id}/status", response_model=ProjectStatus)
def get_status(project_id: str, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    return ProjectStatus(
        project_id=str(project.id),
        status=project.status,
        created_at=str(project.created_at),
    )


# ── POST /projects/{id}/mint ───────────────────────────────────────────────────
@router.post("/{project_id}/mint")
def mint_credits(project_id: str, db: Session = Depends(get_db)):
    """
    Mints verified carbon credits as ERC-20 tokens on Sepolia testnet.

    Flow:
    1. Validates project is in 'passed' status
    2. Finds the latest unminted PASS verification
    3. Checks for duplicate / already-recorded mint attempts
    4. Reads dynamic gas fees from the network
    5. Calls mintCredits() on the deployed CarbonCredit contract
    6. Waits for confirmation, saves tx_hash to DB
    7. Returns tx_hash and Etherscan link
    """
    private_key = _require_env("BLOCKCHAIN_PRIVATE_KEY", PRIVATE_KEY)

    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if project.status != "passed":
        raise HTTPException(
            status_code=400,
            detail=(
                f"Cannot mint — project status is '{project.status}'. "
                f"Must be 'passed'. Flagged projects need review clearance first."
            ),
        )

    verification = (
        db.query(Verification)
        .filter(
            Verification.project_id == project_id,
            Verification.decision == "PASS",
        )
        .order_by(Verification.created_at.desc())
        .first()
    )

    if not verification:
        raise HTTPException(
            status_code=404,
            detail="No PASS verification found for this project",
        )

    # If the DB already knows this verification was minted, do not resend the tx.
    if _verification_already_persisted(db, verification):
        existing_tx = getattr(verification, "tx_hash", None)
        return {
            "project_id": str(project.id),
            "company_name": project.company_name,
            "status": "credits_issued",
            "year_number": verification.year_number,
            "active_credits": verification.active_credits,
            "buffer_credits": verification.buffer_credits,
            "tx_hash": existing_tx,
            "etherscan": f"{SEPOLIA_EXPLORER_BASE}{existing_tx}" if existing_tx else None,
            "message": "This verification is already minted. No new transaction was sent.",
        }

    if not verification.active_credits or verification.active_credits <= 0:
        raise HTTPException(status_code=400, detail="Active credits are zero — nothing to mint")

    try:
        w3 = _get_w3()
        contract = _get_contract(w3)
        account = w3.eth.account.from_key(private_key)

        year_number = _as_int(verification.year_number, 1)
        verification_id_for_chain = _build_verification_key(
            str(project.id),
            year_number,
            str(verification.id),
        )

        # Best-effort duplicate protection: if the contract exposes a getter,
        # try to detect previous mint records before broadcasting.
        already_recorded_key = _try_onchain_duplicate_check(
            contract=contract,
            project_id=str(project.id),
            year_number=year_number,
            verification_id=verification_id_for_chain,
        )
        if already_recorded_key:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Verification already recorded on-chain. "
                    f"Detected key: {already_recorded_key}"
                ),
            )

        image_hash_str = _extract_primary_image_hash(verification.image_hashes)
        ndvi_basis_points = int(round(_as_float(verification.ndvi_current) * 10000))

        # Estimate gas first; if the contract wants to revert, fail before we spend gas.
        tx_args: Dict[str, Any] = {
            "from": account.address,
            "nonce": w3.eth.get_transaction_count(account.address, "pending"),
            "chainId": 11155111,  # Sepolia
        }
        tx_args.update(_build_fee_params(w3))

        gas_estimate = contract.functions.mintCredits(
            str(project.id),
            verification_id_for_chain,
            image_hash_str,
            Web3.to_checksum_address(project.company_wallet),
            ndvi_basis_points,
            _as_int(verification.confidence_score),
            _as_int(verification.active_credits),
            _as_int(verification.buffer_credits),
        ).estimate_gas(tx_args)

        gas_limit = int(max(gas_estimate * 13 // 10, 300000))
        gas_limit = min(gas_limit, 900000)
        tx_args["gas"] = gas_limit

        tx = contract.functions.mintCredits(
            str(project.id),
            verification_id_for_chain,
            image_hash_str,
            Web3.to_checksum_address(project.company_wallet),
            ndvi_basis_points,
            _as_int(verification.confidence_score),
            _as_int(verification.active_credits),
            _as_int(verification.buffer_credits),
        ).build_transaction(tx_args)

        project.status = "minting"
        db.commit()

        signed_tx = w3.eth.account.sign_transaction(tx, private_key)
        tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        tx_hash_hex = tx_hash.hex()

        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=180)

        if receipt["status"] != 1:
            project.status = "passed"
            db.commit()
            raise HTTPException(
                status_code=409,
                detail=f"Transaction reverted on-chain. Check: {SEPOLIA_EXPLORER_BASE}{tx_hash_hex}",
            )

        verification.tx_hash = tx_hash_hex
        if hasattr(Verification, "minted_at"):
            verification.minted_at = dt.datetime.utcnow()

        project.status = "credits_issued"
        db.commit()

        return {
            "project_id": str(project.id),
            "company_name": project.company_name,
            "status": "credits_issued",
            "year_number": verification.year_number,
            "active_credits": verification.active_credits,
            "buffer_credits": verification.buffer_credits,
            "tx_hash": tx_hash_hex,
            "etherscan": f"{SEPOLIA_EXPLORER_BASE}{tx_hash_hex}",
        }

    except HTTPException:
        raise

    except Exception as e:
        # If anything fails mid-mint, keep the project usable for a retry.
        project.status = "passed"
        db.commit()
        raise HTTPException(status_code=500, detail=f"Mint failed: {str(e)}")


# ── GET /projects/{id}/report ──────────────────────────────────────────────────
@router.get("/{project_id}/report")
def get_report(project_id: str, db: Session = Depends(get_db)):
    """
    Full audit report for government submission or credit buyer due diligence.
    Includes all image hashes, NDVI values, confidence score, tx_hash, Etherscan link.
    """
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    verifications = (
        db.query(Verification)
        .filter(Verification.project_id == project_id)
        .order_by(Verification.year_number.asc())
        .all()
    )

    return {
        "project_id": str(project.id),
        "company_name": project.company_name,
        "company_wallet": project.company_wallet,
        "area_hectares": project.area_hectares,
        "plantation_date": project.plantation_date,
        "status": project.status,
        "verifications": [
            {
                "year": v.year_number,
                "decision": v.decision,
                "confidence_score": v.confidence_score,
                "active_credits": v.active_credits,
                "buffer_credits": v.buffer_credits,
                "tree_cover_pct": v.tree_cover_pct,
                "ndvi_baseline": v.ndvi_baseline,
                "ndvi_current": v.ndvi_current,
                "image_hashes": v.image_hashes,
                "tx_hash": v.tx_hash,
                "etherscan": f"{SEPOLIA_EXPLORER_BASE}{v.tx_hash}" if v.tx_hash else None,
                "minted_at": str(v.minted_at) if getattr(v, "minted_at", None) else None,
                "verified_at": str(v.created_at),
            }
            for v in verifications
        ],
    }


# ─────────────────────────────────────────────────────────────
# GET ALL PROJECTS
# Used by Admin Dashboard frontend
# ─────────────────────────────────────────────────────────────

@router.get("/")
def get_all_projects(
    wallet: str = Query(default=None, description="Filter by company wallet address"),
    db: Session = Depends(get_db)
):
    """
    Returns all projects for admin (no wallet param).
    Returns only that company's projects when wallet param is provided.
    This is how Vantara sees only their submissions, not everyone else's.
    """
    query = db.query(Project)

    if wallet:
        # Case-insensitive wallet match — MetaMask sometimes changes capitalisation
        query = query.filter(
            Project.company_wallet.ilike(wallet)
        )

    projects = query.order_by(Project.created_at.desc()).all()

    return [
        {
            "project_id"     : str(p.id),
            "company_name"   : p.company_name,
            "company_wallet" : p.company_wallet,
            "area_hectares"  : p.area_hectares,
            "plantation_date": p.plantation_date,
            "status"         : p.status,
            "created_at"     : str(p.created_at)
        }
        for p in projects
    ]