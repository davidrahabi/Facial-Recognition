// Store webcam streams
let streams = {1: null, 2: null};

// Add event listeners for file inputs
document.addEventListener('DOMContentLoaded', function() {
    // Handle file uploads
    document.getElementById('file1').addEventListener('change', (e) => handleFileSelect(e, 1));
    document.getElementById('file2').addEventListener('change', (e) => handleFileSelect(e, 2));
});

/**
 * Handle file selection for image upload
 * @param {Event} event - The file input change event
 * @param {number} number - The image number (1 or 2)
 */
function handleFileSelect(event, number) {
    const file = event.target.files[0];
    const canvas = document.getElementById(`canvas${number}`);
    const ctx = canvas.getContext('2d');

    const img = new Image();
    img.onload = () => {
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);
    };
    img.src = URL.createObjectURL(file);
}

/**
 * Start webcam for image capture
 * @param {number} number - The image number (1 or 2)
 */
function startCamera(number) {
    const video = document.getElementById(`webcam${number}`);
    video.style.display = 'block';
    
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            streams[number] = stream;
            video.srcObject = stream;
        })
        .catch(error => {
            console.error('Error accessing webcam:', error);
            alert('Unable to access webcam. Please ensure you have granted camera permissions.');
        });
}

/**
 * Capture image from webcam
 * @param {number} number - The image number (1 or 2)
 */
function captureImage(number) {
    const video = document.getElementById(`webcam${number}`);
    const canvas = document.getElementById(`canvas${number}`);
    const context = canvas.getContext('2d');

    // Make sure video is playing
    if (!video.srcObject || !streams[number]) {
        alert('Please start the camera first.');
        return;
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0);

    // Stop the camera
    if (streams[number]) {
        streams[number].getTracks().forEach(track => track.stop());
        streams[number] = null;
    }
    video.style.display = 'none';
}

/**
 * Handle consent acceptance
 */
function acceptConsent() {
    if (!document.getElementById('consentCheckbox').checked || 
        !document.getElementById('ageVerification').checked) {
        alert('Please accept the privacy policy and verify your age');
        return;
    }
    // Hide the modal
    document.getElementById('consentModal').style.display = 'none';
}

/**
 * Compare the two images for face verification
 */
function compareImages() {
    if (!document.getElementById('consentCheckbox').checked || 
        !document.getElementById('ageVerification').checked) {
        alert('Please accept the privacy policy and verify your age');
        return;
    }

    const canvas1 = document.getElementById('canvas1');
    const canvas2 = document.getElementById('canvas2');
    const resultElement = document.getElementById('result');

    // Check if images are loaded
    const ctx1 = canvas1.getContext('2d');
    const ctx2 = canvas2.getContext('2d');
    
    // Simple check to see if canvases have content
    const imageData1 = ctx1.getImageData(0, 0, canvas1.width, canvas1.height);
    const imageData2 = ctx2.getImageData(0, 0, canvas2.width, canvas2.height);
    
    if (!hasImageData(imageData1) || !hasImageData(imageData2)) {
        resultElement.textContent = 'Please load or capture both images first';
        resultElement.className = 'result-error';
        return;
    }

    // Reset result styling
    resultElement.className = 'result-neutral';
    
    // Show loading state
    document.getElementById('loading').style.display = 'flex';
    resultElement.textContent = '';

    // Convert canvases to blobs
    Promise.all([
        new Promise(resolve => canvas1.toBlob(resolve, 'image/jpeg')),
        new Promise(resolve => canvas2.toBlob(resolve, 'image/jpeg'))
    ]).then(blobs => {
        const formData = new FormData();
        formData.append('image1', blobs[0]);
        formData.append('image2', blobs[1]);

        fetch('/verify', {
            method: 'POST',
            headers: {
                'user-consent': 'true',
                'age-verified': 'true'
            },
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            // Hide loading state
            document.getElementById('loading').style.display = 'none';
            
            if (data.error) {
                resultElement.textContent = `Error: ${data.error}`;
                resultElement.className = 'result-error';
            } else {
                const similarityPercentage = (data.similarity_score * 100).toFixed(2);
                const resultText = data.verified ? 
                    `✓ Match Confirmed: Same Person` : 
                    `✗ No Match: Different Persons`;
                    
                resultElement.innerHTML = `${resultText} <br><span class="result-percentage">Similarity: ${similarityPercentage}%</span>`;
                resultElement.className = data.verified ? 'result-success' : 'result-error';
            }
        })
        .catch(error => {
            // Hide loading and show error
            document.getElementById('loading').style.display = 'none';
            resultElement.textContent = `Error: ${error.message}`;
            resultElement.className = 'result-error';
        });
    });
}

/**
 * Helper function to check if ImageData contains actual image content
 * @param {ImageData} imageData - The ImageData to check
 * @return {boolean} - True if there is image data
 */
function hasImageData(imageData) {
    // Check if the image data contains non-transparent pixels
    const data = imageData.data;
    // Check a sample of pixels
    for (let i = 0; i < data.length; i += 40) {
        // If any pixel has a non-zero value, assume there's an image
        if (data[i] !== 0 || data[i+1] !== 0 || data[i+2] !== 0 || data[i+3] !== 0) {
            return true;
        }
    }
    return false;
}