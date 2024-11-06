from flask import Flask, request, jsonify
from data_manager import DataManager
from rag_model import RagModel
from chain_of_thought import ChainOfThought
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize components
try:
    data_manager = DataManager()
    rag_model = RagModel(data_manager)
    chain_of_thought = ChainOfThought(data_manager, rag_model)
    logger.info("Successfully initialized all components")
except Exception as e:
    logger.error(f"Error during initialization: {str(e)}")
    raise


@app.route('/')
def home():
    return "Chatbot API is running. Use /chatbot for queries and /update_embeddings to update content."


@app.route('/chatbot', methods=['POST'])
def chatbot():
    try:
        logger.debug(f"Received request: {request.get_json()}")
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No JSON data received'}), 400

        if 'user_query' not in data:
            return jsonify({'error': 'user_query is required'}), 400

        user_query = data['user_query']
        previous_context = data.get('previous_context', None)

        response = chain_of_thought.generate_response(user_query, previous_context)
        logger.debug(f"Generated response: {response}")

        return jsonify({'response': response})
    except Exception as e:
        logger.error(f"Error in chatbot endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/update_embeddings', methods=['POST'])
def update_embeddings_endpoint():
    try:
        logger.debug(f"Received update request: {request.get_json()}")
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No JSON data received'}), 400

        if 'post_id' not in data or 'post_text' not in data:
            return jsonify({'error': 'post_id and post_text are required'}), 400

        post_id = data['post_id']
        post_text = data['post_text']

        data_manager.update_embeddings(post_id, post_text)
        data_manager.save_index()

        return jsonify({'status': 'success', 'message': f'Successfully updated embeddings for post {post_id}'})
    except Exception as e:
        logger.error(f"Error in update_embeddings endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)