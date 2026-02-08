"""Schema definitions for the Pharmacy Assistant."""

from typing import Annotated, List, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel


class Price(BaseModel):
    """Price information for a medication."""

    amount: float
    currency: str


class BaseInfo(BaseModel):
    """Basic product information for a medication."""

    form: str
    brand: str
    price: Price
    pack_size: str


class Stock(BaseModel):
    """Stock availability information."""

    status: str
    location: str
    quantity: int


class Regulatory(BaseModel):
    """Regulatory and prescription requirements."""

    narcotic: bool
    age_restriction: int
    requires_prescription: bool


class SafetyWarnings(BaseModel):
    """Safety warnings and contraindications."""

    contraindications: List[str]
    side_effects_summary: str


class UsageInstructions(BaseModel):
    """Dosage and administration instructions."""

    dosage: str
    administration: str


class MedicalData(BaseModel):
    """Medical usage data and guidelines."""

    safety_warnings: SafetyWarnings
    application_area: List[str]
    active_ingredients: List[str]
    usage_instructions: UsageInstructions


class Medication(BaseModel):
    """
    Represents a medication from the database.

    Attributes:
        id: Unique identifier of the medication.
        name: Name of the medication.
        base_info: Basic product information (form, brand, price, pack size).
        stock: Stock availability information (status, location, quantity).
        regulatory: Regulatory information (prescription requirements, age restrictions).
        medical_data: Medical usage data (warnings, application areas, ingredients, dosage).
    """

    id: int
    name: str
    base_info: BaseInfo
    stock: Stock
    regulatory: Regulatory
    medical_data: MedicalData


class AgentState(TypedDict):
    """
    State schema for the pharmacy assistant agent.

    Attributes:
        messages: Conversation history with automatic message aggregation.
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]
