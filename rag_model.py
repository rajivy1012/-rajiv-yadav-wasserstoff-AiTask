# File: rag_model.py
from transformers import AutoTokenizer, AutoModel
import torch


class RagModel:
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large')
        self.model = AutoModel.from_pretrained('facebook/bart-large')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

    def generate_response(self, user_query, previous_context=None):
        # Get relevant passages using FAISS
        query_embedding = self.data_manager.embed_model.encode([user_query])
        distances, indices = self.data_manager.search_index(query_embedding)

        # Get relevant passages
        relevant_passages = []
        for idx in indices[0]:
            passage = self.data_manager.get_post_text(str(idx))
            if passage != "Post not found":
                relevant_passages.append(passage)

        # Combine query with context
        context = " ".join(relevant_passages)
        input_text = f"Query: {user_query}\nContext: {context}"

        # Tokenize and generate
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=1024,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=128,
                num_beams=4,
                no_repeat_ngram_size=3,
                top_k=50,
                top_p=0.95,
                do_sample=True
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._format_response(response, relevant_passages)

    def _format_response(self, response, relevant_passages):
        return {
            'response': response,
            'sources': [passage[:200] + '...' for passage in relevant_passages]
        }