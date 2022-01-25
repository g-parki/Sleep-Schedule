#Ensure project folder is added to path
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.path[0]).parent))

#Initialize database and app
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db_abs_path = os.path.realpath(os.path.join(Path(__file__).parent.parent, 'data', 'site.db'))

app = Flask(__name__)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + db_abs_path
db = SQLAlchemy(app)

demo_date = datetime(year=2021, month=7, day=9, hour=4, minute=0)

from scripts import routes
