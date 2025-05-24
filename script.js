document.getElementById('predictionForm').addEventListener('submit', function(e) {
    e.preventDefault();
    
    // Collect form data
    const formData = {};
    const inputs = document.querySelectorAll('input[type="number"]');
    inputs.forEach(input => {
        formData[input.name] = parseFloat(input.value);
    });
    
    // Make API call
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(formData)
    })
    .then(response => response.json())
    .then(data => {
        const resultDiv = document.getElementById('predictionResult');
        const predictionText = document.getElementById('predictionText');
        const probabilityText = document.getElementById('probabilityText');
        
        if (data.error) {
            resultDiv.style.display = 'block';
            resultDiv.className = 'prediction-box';
            predictionText.textContent = `Error: ${data.error}`;
            probabilityText.textContent = '';
        } else {
            resultDiv.style.display = 'block';
            if (data.prediction === 'M') {
                resultDiv.className = 'prediction-box prediction-malignant';
                predictionText.textContent = 'Prediction: Malignant';
            } else {
                resultDiv.className = 'prediction-box prediction-benign';
                predictionText.textContent = 'Prediction: Benign';
            }
            probabilityText.textContent = `Probability: ${data.probability}%`;
        }
    })
    .catch(error => {
        const resultDiv = document.getElementById('predictionResult');
        const predictionText = document.getElementById('predictionText');
        resultDiv.style.display = 'block';
        resultDiv.className = 'prediction-box';
        predictionText.textContent = `Error: ${error.message}`;
        document.getElementById('probabilityText').textContent = '';
    });
}); 