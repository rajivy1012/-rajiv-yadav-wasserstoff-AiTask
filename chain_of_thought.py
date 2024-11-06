# File: chain_of_thought.py
class ChainOfThought:
    def __init__(self, data_manager, rag_model):
        self.data_manager = data_manager
        self.rag_model = rag_model

    def generate_response(self, user_query, previous_context=None):
        # Get response from RAG model
        rag_response = self.rag_model.generate_response(user_query, previous_context)

        # Build chain of thought response
        response = {
            'initial_response': rag_response['response'],
            'chain_of_thought': [
                f"Query: {user_query}",
                "Analyzing relevant sources:",
            ],
            'sources': rag_response['sources'],
            'final_response': rag_response['response']
        }

        return response