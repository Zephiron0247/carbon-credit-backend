from sqlalchemy import Column, String, Float, Integer, DateTime, Boolean, JSON, ForeignKey, Enum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from database import Base
import uuid
import datetime
import enum

'''class ProjectStatus(enum.Enum):
    pending    = "pending"
    verifying  = "verifying"
    passed     = "passed"
    failed     = "failed"
'''

class Project(Base):
    __tablename__ = "projects"

    id               = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    company_wallet   = Column(String, nullable=False)   # blockchain wallet address
    company_name     = Column(String, nullable=False)
    coordinates      = Column(JSON, nullable=False)      # list of [lon, lat] pairs
    area_hectares    = Column(Float, nullable=False)
    plantation_date  = Column(String, nullable=False)    # when trees were planted
    status           = Column(String, default="pending")
    created_at       = Column(DateTime, default=datetime.datetime.utcnow)

    # One project has many verification cycles (year 1, year 2, etc.)
    verifications    = relationship("Verification", back_populates="project")

class Verification(Base):
    __tablename__ = "verifications"

    id               = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id       = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False)
    year_number      = Column(Integer, nullable=False)   # which annual cycle this is
    ndvi_baseline    = Column(Float)
    ndvi_current     = Column(Float)
    stage1_credits   = Column(Integer)
    tree_cover_pct   = Column(Float)
    confidence_score = Column(Float)
    decision         = Column(String)                    # PASS or FAIL
    adjusted_credits = Column(Integer)
    buffer_credits   = Column(Integer)
    active_credits   = Column(Integer)
    image_hashes     = Column(JSON)                      # list of hashes per checkpoint
    tx_hash          = Column(String, nullable=True)     # filled after blockchain mint
    minted_at = Column(DateTime, nullable=True)  # filled when blockchain mint succeeds
    created_at       = Column(DateTime, default=datetime.datetime.utcnow)

    project          = relationship("Project", back_populates="verifications")

class CreditLedger(Base):
    __tablename__ = "credit_ledger"

    id               = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id       = Column(UUID(as_uuid=True), ForeignKey("projects.id"))
    verification_id  = Column(UUID(as_uuid=True), ForeignKey("verifications.id"))
    credits_issued   = Column(Integer)
    credits_buffer   = Column(Integer)
    credits_active   = Column(Integer)
    tx_hash          = Column(String, nullable=True)
    issued_at        = Column(DateTime, default=datetime.datetime.utcnow)

class FraudFlag(Base):
    __tablename__ = "fraud_flags"

    id               = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id       = Column(UUID(as_uuid=True), ForeignKey("projects.id"))
    flag_type        = Column(String)    # 'ndvi_crash', 'low_confidence', 'spatial_anomaly'
    details          = Column(JSON)
    flagged_at       = Column(DateTime, default=datetime.datetime.utcnow)
    resolved         = Column(Boolean, default=False)