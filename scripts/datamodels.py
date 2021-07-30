import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.path[0]).parent))

from datetime import datetime
from scripts import db


class DataPoint(db.Model):
    """Model class for a classified image with timestamp"""
    id = db.Column(db.Integer, primary_key= True)
    timestamp = db.Column(db.DateTime, default= datetime.utcnow)
    value = db.Column(db.String, nullable= False)
    baby_reading = db.Column(db.Float, nullable= False)
    empty_reading = db.Column(db.Float, nullable= False)
    image_orig_path = db.Column(db.String, nullable= False)
    image_resized_path = db.Column(db.String, nullable= False)
    in_training_data = db.Column(db.Boolean, nullable= True)

    def __repr__(self) -> str:
        return (f'ID: {self.id}\n'
            f'Time: {self.timestamp}\n'
            f'Value: {self.value}\n'
            f'Baby Reading: {self.baby_reading}\n'
            f'Empty Reading: {self.baby_reading}\n'
            f'Image orig path: {self.image_orig_path}\n'
            f'Image resized path: {self.image_resized_path}\n'
            f'In training data: {self.in_training_data}'
        )

def commit_item(model_obj):
    db.session.add(model_obj)
    db.session.commit()
