from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from chatbot import ChatBot

app = Flask(__name__)
CORS(app)  # Enable CORS for the entire app (allows requests from all domains)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        # Get the user input from the request
        user_input = request.json.get("message")
        
        if not user_input:
            return jsonify({'error': 'No input provided'}), 400
        
        # Call the AskQuestion function from chatbot.py
        response = k.ask_question(user_input)

        # Check if there is an error in the response
        if "error" in response:
            print(f"Error in chatbot response: {response['error']}")
            return jsonify({'error': response["error"]}), 500
        
        # Only return the AI response (ignore related_Doc)
        ai_response = response.get("AI")
        
        if ai_response:
            return jsonify({"AI": ai_response})
        else:
            return jsonify({'error': 'AI response not found'}), 500

    except Exception as e:
        print(f"Error in chat route: {e}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Start the Flask app on the default port (5000)
    k =  ChatBot(csv_path=r"D:\ChatBot\python flask server\JNTUH_Student_Services_FAQ_updated.csv",
        api_key="AIzaSyBAVrawMMPMRmUaXXV-mk-ujZpsbSZbtTQ",
        model_name="gemini-1.5-pro"
    )
    app.run(debug=True)
