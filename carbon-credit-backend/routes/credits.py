import os
import json
import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from web3 import Web3
from pydantic import BaseModel

from database import get_db
from models import Project, Verification

router = APIRouter(prefix="/credits", tags=["credits"])

CONTRACT_ADDRESS = "0x8d0B4Dbd29ae0a52C1B3d2B4568DFE6aF1032285"
SEPOLIA_RPC_URL  = os.getenv("SEPOLIA_RPC_URL", "https://rpc.sepolia.org")

_ABI_PATH = os.path.join(os.path.dirname(__file__), "..", "CarbonCredit.json")
with open(_ABI_PATH, "r") as _f:
    CONTRACT_ABI = json.load(_f)["abi"]


def _get_w3():
    w3 = Web3(Web3.HTTPProvider(SEPOLIA_RPC_URL))
    if not w3.is_connected():
        raise ConnectionError("Cannot connect to Sepolia")
    return w3


def _get_contract(w3):
    return w3.eth.contract(
        address=Web3.to_checksum_address(CONTRACT_ADDRESS),
        abi=CONTRACT_ABI
    )


@router.get("/available")
def get_available_credits(db: Session = Depends(get_db)):
    """
    Returns all projects with credits_issued status for the buyer marketplace.
    Includes satellite verification data so buyers can audit what they're buying.
    """
    projects = db.query(Project).filter(
        Project.status == "credits_issued"
    ).all()

    result = []
    for p in projects:
        verification = (
            db.query(Verification)
            .filter(
                Verification.project_id == p.id,
                Verification.tx_hash != None
            )
            .order_by(Verification.created_at.desc())
            .first()
        )

        if not verification:
            continue

        result.append({
            "project_id"      : str(p.id),
            "company_name"    : p.company_name,
            "company_wallet"  : p.company_wallet,
            "area_hectares"   : p.area_hectares,
            "plantation_date" : p.plantation_date,
            "active_credits"  : verification.active_credits,
            "buffer_credits"  : verification.buffer_credits,
            "confidence_score": verification.confidence_score,
            "ndvi_current"    : verification.ndvi_current,
            "tree_cover_pct"  : verification.tree_cover_pct,
            "tx_hash"         : verification.tx_hash,
            "verification_id" : str(verification.id),
            "etherscan"       : f"https://sepolia.etherscan.io/tx/{verification.tx_hash}"
        })

    return result


class RetireRequest(BaseModel):
    project_id        : str
    verification_id   : str
    amount            : int
    buyer_wallet      : str
    buyer_private_key : str   # used locally to sign, never stored


@router.post("/retire")
def retire_credits(payload: RetireRequest, db: Session = Depends(get_db)):
    """
    Called when a buyer retires credits to offset their emissions.
    Burns tokens permanently on-chain — irreversible proof of carbon offset.
    """
    try:
        w3       = _get_w3()
        contract = _get_contract(w3)
        account  = w3.eth.account.from_key(payload.buyer_private_key)

        if account.address.lower() != payload.buyer_wallet.lower():
            raise HTTPException(
                status_code=400,
                detail="Buyer wallet address does not match private key"
            )

        balance = contract.functions.balanceOf(
            Web3.to_checksum_address(payload.buyer_wallet)
        ).call()

        if balance < payload.amount:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient credits. Have {balance}, need {payload.amount}"
            )

        fee_history = w3.eth.fee_history(1, 'latest', [50])
        base_fee    = fee_history['baseFeePerGas'][-1]
        priority    = w3.to_wei('2', 'gwei')
        max_fee     = base_fee * 2 + priority

        tx = contract.functions.retireCredits(
            payload.amount,
            payload.verification_id,
            payload.project_id
        ).build_transaction({
            'from'                 : account.address,
            'nonce'                : w3.eth.get_transaction_count(account.address, 'latest'),
            'gas'                  : 200000,
            'chainId'              : 11155111,
            'maxFeePerGas'         : max_fee,
            'maxPriorityFeePerGas' : priority,
        })

        signed  = account.sign_transaction(tx)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

        if receipt.status != 1:
            raise Exception("Retire transaction reverted on-chain")

        tx_hash_hex = tx_hash.hex()

        return {
            "status"          : "retired",
            "amount_retired"  : payload.amount,
            "buyer_wallet"    : payload.buyer_wallet,
            "project_id"      : payload.project_id,
            "verification_id" : payload.verification_id,
            "tx_hash"         : tx_hash_hex,
            "etherscan"       : f"https://sepolia.etherscan.io/tx/{tx_hash_hex}",
            "retired_at"      : str(datetime.datetime.utcnow())
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retire failed: {str(e)}")