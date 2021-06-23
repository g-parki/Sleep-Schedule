#Ensure project folder is added to path
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.path[0]).parent))

#Initialize database and app
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

db_abs_path = os.path.realpath(os.path.join(Path(__file__).parent.parent, 'data', 'site.db'))

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + db_abs_path
db = SQLAlchemy(app)

from scripts import routes
