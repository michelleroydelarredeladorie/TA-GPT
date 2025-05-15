from flask import Flask, request, jsonify, render_template, session, redirect, g
from flask_session import Session as FlaskSession
from flask_sqlalchemy import SQLAlchemy
import os
from dotenv import load_dotenv
from datetime import datetime
import json
from helpers import ask, followup, init_clients, load_embeddings, login_required, route_ask

# Global variables
vo = None
client = None

# Load environmental variables
load_dotenv()
init_clients()
document_names, documents, embeddings , metadata = load_embeddings()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY is not set in the environment!")

VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
if not VOYAGE_API_KEY:
    raise ValueError("VOYAGE_API_KEY is not set in the environment!")

FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY")
if not FLASK_SECRET_KEY:
    raise ValueError("FLASK_SECRET_KEY is not set in the environment!")

####################################################
app = Flask(__name__)

# Ensure templates are auto-reloaded
app.config["TEMPLATES_AUTO_RELOAD"] = True

# Determine the absolute directory path of the current file (application.py)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Build the full absolute path to TA_GPT.db
DATABASE_PATH = os.path.join(BASE_DIR, "TA_GPT.db")
print("Database path:", DATABASE_PATH)

# Configure SQLAlchemy with the absolute path
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{DATABASE_PATH}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = FLASK_SECRET_KEY

# Configure server-side session storage
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = './flask_sessions'
app.config['SESSION_PERMANENT'] = False
FlaskSession(app)

db = SQLAlchemy(app)

##############################################
# DATABASE MODELS
##############################################

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)

class ChatSession(db.Model):
    __tablename__ = 'sessions'
    session_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    start_time = db.Column(db.DateTime, nullable=False)
    end_time = db.Column(db.DateTime, nullable=True)

class Question(db.Model):
    __tablename__ = 'questions'
    question_id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('sessions.session_id'), nullable=False)
    question = db.Column(db.Text, nullable=False)
    context = db.Column(db.Text, nullable=False)  # Can store JSON as text
    time = db.Column(db.DateTime, nullable=False)

class Answer(db.Model):
    __tablename__ = 'answers'
    answer_id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('sessions.session_id'), nullable=False)
    answer = db.Column(db.Text, nullable=False)
    sources = db.Column(db.Text, nullable=False)
    # Now allow feedback to be None (no feedback), True (thumbs up), or False (thumbs down)
    feedback = db.Column(db.Boolean, nullable=True)
    time = db.Column(db.DateTime, nullable=False)

with app.app_context():
    db.create_all()
    users = User.query.all()
    print("Existing users:", users)

##############################################
# ROUTES
##############################################

@app.route('/login', methods=['GET', 'POST'])
def login():
    session.clear()

    if request.method == 'POST':
        data = request.get_json()  # Expecting JSON data
        user_id = str(data.get('id'))

        if not user_id:
            return jsonify({'message': 'User ID is required!'}), 400

        # Use the new Session.get() method instead of Query.get()
        user = db.session.get(User, user_id)
        if user:
            # Create a new ChatSession record
            new_session = ChatSession(
                user_id=user.id,
                start_time=datetime.now(),
                end_time=None
            )
            db.session.add(new_session)
            db.session.commit()

            # Save user info and chat_session_id in Flask session
            session['user_id'] = user.id
            session['user_name'] = user.name
            session['chat_session_id'] = new_session.session_id

            return redirect("/")
        else:
            return jsonify({'message': 'User not found!'}), 404

    # If GET request, serve the login page
    return render_template('login_page.html')

@app.route('/', methods=['GET'])
@login_required
def home():
    session['is_first_question'] = True
    user_name = session.get('user_name', 'Guest')
    return render_template('chatbot_clean.html', user_name=user_name)

@app.route('/chat', methods=['POST'])
@login_required
def chat():
    data = request.json
    user_message = data.get('message', '')
    images = data.get('images', [])
    source_type = data.get('source_type', 'Default')
    chat_session_id = session.get('chat_session_id')
    user_id = session.get('user_id')

    # Check if user should see Version A or B (Version A doesn't send source_type)
    ab_version = 'B' if int(user_id) % 2 != 0 else 'A'

    # Create a JSON representation of the context that includes the images
    context_data = {
        'text': user_message,
        'has_images': len(images) > 0,
        'image_count': len(images),
        'source_type': source_type if ab_version == 'B' else 'Default',
        'ab_version': ab_version
    }

    # Log the question in the database
    question_obj = Question(
        session_id=chat_session_id,
        question=user_message,
        context=json.dumps(context_data),  # Store context as JSON
        time=datetime.now()
    )
    db.session.add(question_obj)
    db.session.commit()

    # Get the bot response using ask() or followup() based on conversation state
    is_first_question = session.get('is_first_question', True)

    if ab_version == 'B':
        # For Version B, use the functions that accept source_type
        if is_first_question:
            bot_response, sources = route_ask(user_message, source_type, images)
            session['is_first_question'] = False
        else:
            bot_response, sources = followup(user_message, source_type, images)
    else:
        # For Version A, use the original functions without source_type
        if is_first_question:
            bot_response, sources = route_ask(user_message, "Default", images)
            session['is_first_question'] = False
        else:
            bot_response, sources = followup(user_message, "Default", images)

    # Log the answer in the database with default feedback value as None (meaning no feedback yet)
    answer_obj = Answer(
        session_id=chat_session_id,
        answer=bot_response,
        sources=sources,
        feedback=None,  # No feedback provided yet
        time=datetime.now()
    )
    db.session.add(answer_obj)
    db.session.commit()

    return jsonify({'response': bot_response})


@app.route('/feedback', methods=['POST'])
@login_required
def feedback():
    """
    Receives feedback for a bot answer from the front end.
    Expects a JSON payload with:
      - message: the answer text from the bot
      - feedback: boolean value (True for thumbs up, False for thumbs down)
    Finds the corresponding Answer record (by matching session and answer text)
    and updates its feedback value.
    """
    data = request.get_json()
    message_text = data.get('message')
    feedback_value = data.get('feedback')
    chat_session_id = session.get('chat_session_id')
    
    if chat_session_id is None:
        return jsonify({'message': 'No active chat session.'}), 400

    # Find the latest answer record in this session with matching answer text
    answer_record = Answer.query.filter_by(session_id=chat_session_id, answer=message_text)\
                        .order_by(Answer.time.desc()).first()

    if answer_record:
        answer_record.feedback = feedback_value
        db.session.commit()
        return jsonify({'message': 'Feedback updated.'}), 200
    else:
        return jsonify({'message': 'Answer record not found.'}), 404
    
@app.route('/get_ab_version', methods=['GET'])
@login_required
def get_ab_version():
    """Endpoint to determine if user should see version A or B based on user ID"""
    user_id = session.get('user_id')
    # Even user IDs get version A, odd get version B
    version = 'A' if int(user_id) % 2 == 0 else 'B'
    return jsonify({'version': version})

@app.before_request
def require_login():
    if not session.get("user_id") and request.endpoint not in ["login", "static"]:
        return redirect("/login")

@app.route('/logout', methods=['GET'])
def logout():
    chat_session_id = session.get('chat_session_id')
    if chat_session_id:
        # Use the new API to fetch the ChatSession record
        chat_sess = db.session.get(ChatSession, chat_session_id)
        if chat_sess:
            chat_sess.end_time = datetime.now()
            db.session.commit()

    session.clear()
    return redirect('/login')

###################################################
# Initialize the clients and load embeddings
###################################################
if __name__ == '__main__':
    init_clients()
    document_names, documents, embeddings ,  metadata = load_embeddings()
    app.run(debug=True)




