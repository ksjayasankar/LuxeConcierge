from database import db # Absolute import
from sqlalchemy.types import String, Text, Float, Integer

class Product(db.Model):
    __tablename__ = "products"

    id = db.Column(Integer, primary_key=True)
    name = db.Column(String(150), nullable=False)
    description = db.Column(Text)
    price = db.Column(Float, nullable=False)
    leather_type = db.Column(String(50), nullable=False)
    color = db.Column(String(50), nullable=False)
    style = db.Column(String(50), nullable=False)
    hardware = db.Column(String(50))
    occasion = db.Column(String(200))
    craftsmanship_score = db.Column(Float)
    sustainability_score = db.Column(Float)

    def to_dict(self):
        """Converts the Product object to a dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "price": self.price,
            "leather_type": self.leather_type,
            "color": self.color,
            "style": self.style,
            "hardware": self.hardware,
            "occasion": self.occasion,
            "craftsmanship_score": self.craftsmanship_score,
            "sustainability_score": self.sustainability_score
        }

    def __repr__(self):
        return f"<Product {self.id}: {self.name}>"