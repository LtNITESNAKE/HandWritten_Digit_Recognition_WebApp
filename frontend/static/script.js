document.addEventListener('DOMContentLoaded', () => {
    const canvas = document.getElementById('drawingCanvas');
    const ctx = canvas.getContext('2d');
    const clearButton = document.getElementById('clear');
    const predictButton = document.getElementById('predict');
    const predictionElement = document.getElementById('prediction');

    let isDrawing = false;
    let lastX = 0;
    let lastY = 0;

    // Set up the canvas
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 15;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    // Drawing event listeners
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);
    canvas.addEventListener('touchstart', handleTouch);
    canvas.addEventListener('touchmove', handleTouch);
    canvas.addEventListener('touchend', stopDrawing);

    clearButton.addEventListener('click', clearCanvas);
    predictButton.addEventListener('click', makePrediction);

    function startDrawing(e) {
        isDrawing = true;
        [lastX, lastY] = getCoordinates(e);
    }

    function draw(e) {
        if (!isDrawing) return;
        e.preventDefault();

        const [currentX, currentY] = getCoordinates(e);

        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(currentX, currentY);
        ctx.stroke();

        [lastX, lastY] = [currentX, currentY];
    }

    function stopDrawing() {
        isDrawing = false;
    }

    function handleTouch(e) {
        e.preventDefault();
        const touch = e.touches[0];
        const mouseEvent = new MouseEvent(e.type === 'touchstart' ? 'mousedown' : 'mousemove', {
            clientX: touch.clientX,
            clientY: touch.clientY
        });
        canvas.dispatchEvent(mouseEvent);
    }

    function getCoordinates(e) {
        const rect = canvas.getBoundingClientRect();
        return [
            e.clientX - rect.left,
            e.clientY - rect.top
        ];
    }

    function clearCanvas() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        predictionElement.textContent = '';
    }    async function makePrediction() {
        // Create a temporary canvas for preprocessing
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = 28;
        tempCanvas.height = 28;
        const tempCtx = tempCanvas.getContext('2d');

        // Set white background
        tempCtx.fillStyle = 'white';
        tempCtx.fillRect(0, 0, 28, 28);
        
        // Draw the digit in black
        tempCtx.fillStyle = 'black';
        tempCtx.strokeStyle = 'black';
        tempCtx.drawImage(canvas, 0, 0, canvas.width, canvas.height, 0, 0, 28, 28);

        // Get the image data
        const imageData = tempCtx.getImageData(0, 0, 28, 28);
        
        // Convert to grayscale array and normalize (using RGBA values)
        const data = [];
        for (let i = 0; i < imageData.data.length; i += 4) {
            // Convert RGB to grayscale using standard weights
            const grayscale = (imageData.data[i] * 0.299 + 
                             imageData.data[i + 1] * 0.587 + 
                             imageData.data[i + 2] * 0.114);
            // Normalize to [0, 1] and invert (MNIST expects white digits on black background)
            data.push((255 - grayscale) / 255.0);
        }

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ image: data })
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }            const result = await response.json();
            predictionElement.textContent = `Predicted digit: ${result.prediction} (Confidence: ${result.confidence.toFixed(2)}%)`;
        } catch (error) {
            console.error('Error:', error);
            predictionElement.textContent = 'Error making prediction';
        }
    }
});
