from flask import Flask, request, jsonify

from akinator import AkinatorModel

app = Flask(__name__)


def process_request_data(request_data, columns):
    input_data = {key: None for key in columns}
    count = 0

    for key in columns:
        if key not in request_data:
            request_data[key] = None

        if request_data[key] is not None:
            if request_data[key] in ['yes', 'no', 'unknown']:
                input_data[key] = request_data[key]
                count += 1
            else:
                if key == 'gender' and request_data[key] in ['male', 'female']:
                    input_data[key] = request_data[key]
                    count += 1
                else:
                    input_data[key] = None
        else:
            input_data[key] = None

    return input_data, count


@app.route('/', methods=['POST'])
def predict_api():
    model = AkinatorModel('/Users/gclee/Downloads/Akinator/akinator.db')
    params = request.get_json()
    columns = model.label_encoders.keys()
    request_data = params.get('data', {key: None for key in columns})
    input_data, count = process_request_data(request_data, columns)

    question = model.select_best_question(input_data)
    predicted_probs = model.predict_person(input_data)
    top_name_and_prob = model.get_top_n_names_and_probabilities(predicted_probs)

    return jsonify({
        "question": {"question": question, "total": len(columns), "count": count},
        "predict": [{"name": predict[0], "probability": predict[1]} for predict in top_name_and_prob],
        "data": input_data
    })


if __name__ == '__main__':
    app.run(debug=True)
