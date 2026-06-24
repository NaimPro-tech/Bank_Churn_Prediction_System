from pydantic import BaseModel, validator

class CustomerData(BaseModel):
    credit_score: int
    country: str
    gender: str
    age: int
    tenure: int
    balance: float
    products_number: int
    credit_card: int
    active_member: str
    estimated_salary: float

    @validator('active_member', pre=True)
    def convert_to_int(cls, value):
        if isinstance(value, str):
            if value.lower() =='yes':
                return 1
            else:
                return 0
        return 