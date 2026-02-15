from sqlalchemy import Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import relationship
from backend.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True)
    password = Column(String, nullable=False)

    analyses = relationship("Analysis", back_populates="owner")


class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, index=True)
    policy_text = Column(Text)
    detected_language = Column(String)
    risk_level = Column(String)
    dpdp_score = Column(String)

    owner_id = Column(Integer, ForeignKey("users.id"))
    owner = relationship("User", back_populates="analyses")
