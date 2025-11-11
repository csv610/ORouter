
# Example Pydantic models for common use cases
class SentimentAnalysis(BaseModel):
    """Model for sentiment analysis results."""
    text: str
    sentiment: str  # positive, negative, neutral
    confidence: float
    reasoning: str


class EntityExtraction(BaseModel):
    """Model for named entity extraction."""
    text: str
    entities: List[Dict[str, str]]  # [{"type": "PERSON", "value": "John"}, ...]


class Classification(BaseModel):
    """Model for text classification."""
    text: str
    category: str
    confidence: float
    categories_considered: List[str]


class Summary(BaseModel):
    """Model for text summarization."""
    original_length: int
    summary: str
    key_points: List[str]


# Usage examples
if __name__ == "__main__":
    # Initialize client
    client = OpenRouterClient(config=ModelConfig(model="deepseek", temperature=0.3))
    
    # Example 1: Simple structured output
    class Person(BaseModel):
        name: str
        age: int
        occupation: str
    
    person = client.generate_structured(
        user_prompt="Tell me about Marie Curie",
        response_model=Person
    )
    print(f"{person.name}, {person.age} - {person.occupation}")
    
    # Example 2: List generation
    class Task(BaseModel):
        name: str
        priority: str
    
    tasks = client.generate_structured_list(
        user_prompt="Create 3 project tasks",
        item_model=Task
    )
    for task in tasks:
        print(f"- {task.name} ({task.priority})")
    
    # Example 3: Regular text
    response = client.generate_text("What is Python?")
    print(response)
