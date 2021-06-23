import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.path[0]).parent))

from scripts import db
from datetime import datetime

class DataPoint(db.Model):
    """Model class for a classified image with timestamp"""
    id = db.Column(db.Integer, primary_key= True)
    timestamp = db.Column(db.DateTime, default= datetime.utcnow)
    value = db.Column(db.String, nullable= False)
    imagepath = db.Column(db.String, nullable= False)

    def __repr__(self) -> str:
        return f'ID: {self.id}\nTime: {self.timestamp}\nValue: {self.value}\nImagepath: {self.imagepath}'

testitem = DataPoint(value='1.0', imagepath='something')
db.session.add(testitem)
db.session.commit()
