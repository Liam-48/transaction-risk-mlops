
from pydantic import BaseModel, ConfigDict, Field


class ApplicationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Financial and applicant characteristics
    income: float
    name_email_similarity: float
    prev_address_months_count: int
    current_address_months_count: int
    customer_age: int
    days_since_request: float
    intended_balcon_amount: float
    payment_type: str
    zip_count_4w: int

    # Transaction and application velocity
    velocity_6h: float
    velocity_24h: float
    velocity_4w: float
    bank_branch_count_8w: int
    date_of_birth_distinct_emails_4w: int

    # Employment and credit information
    employment_status: str
    credit_risk_score: int
    email_is_free: int
    housing_status: str
    phone_home_valid: int
    phone_mobile_valid: int
    bank_months_count: int
    has_other_cards: int
    proposed_credit_limit: float
    foreign_request: int

    # Application device and session
    source: str
    session_length_in_minutes: float
    device_os: str
    keep_alive_session: int
    device_distinct_emails_8w: int
    device_fraud_count: int


class PredictionResponse(BaseModel):
    fraud_probability: float = Field(ge=0, le=1)
    model: str
