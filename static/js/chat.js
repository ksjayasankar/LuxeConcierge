document.addEventListener('DOMContentLoaded', () => {
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const productRecommendations = document.getElementById('product-recommendations');

    // --- Function to add message to chat ---
    function addMessage(sender, text) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', sender); // 'user' or 'assistant'
        messageDiv.textContent = text; // Use textContent to prevent XSS
        chatMessages.appendChild(messageDiv);
        // Scroll to the bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

     // --- Function to display products ---
    function displayProducts(products) {
        productRecommendations.innerHTML = ''; // Clear previous products
        if (products && products.length > 0) {
            products.forEach(product => {
                const card = document.createElement('div');
                card.classList.add('product-card');
                card.innerHTML = `
                    <h4>${escapeHtml(product.name || 'N/A')}</h4>
                    <p><strong>Style:</strong> ${escapeHtml(product.style || 'N/A')}</p>
                    <p><strong>Color:</strong> ${escapeHtml(product.color || 'N/A')}</p>
                    <p><strong>Price:</strong> $${escapeHtml(product.price?.toFixed(2) || 'N/A')}</p>
                    <p>${escapeHtml(product.description?.substring(0, 100) || '')}...</p>
                `;
                productRecommendations.appendChild(card);
            });
            productRecommendations.style.display = 'block'; // Show the section
        } else {
            productRecommendations.style.display = 'none'; // Hide if no products
        }
        // Scroll chat to bottom after potentially adding product section height
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

     // --- Function to show typing indicator ---
    let typingIndicator = null;
    function showTypingIndicator() {
        if (!typingIndicator) {
             typingIndicator = document.createElement('div');
             typingIndicator.classList.add('message', 'assistant', 'typing-indicator');
             typingIndicator.textContent = 'Assistant is typing...';
             chatMessages.appendChild(typingIndicator);
             chatMessages.scrollTop = chatMessages.scrollHeight;
        }
    }

     // --- Function to remove typing indicator ---
    function removeTypingIndicator() {
        if (typingIndicator) {
             chatMessages.removeChild(typingIndicator);
             typingIndicator = null;
        }
    }

    // --- Helper to escape HTML ---
    function escapeHtml(unsafe) {
        if (typeof unsafe !== 'string') {
             // Handle numbers, null, undefined safely
            return unsafe !== null && unsafe !== undefined ? String(unsafe) : '';
        }
        return unsafe
             .replace(/&/g, "&")
             .replace(/</g, "<")
             .replace(/>/g, ">")
             .replace(/"/g, '"')
             .replace(/'/g, "'");
     }


    // --- Send message function ---
    async function sendMessage() {
        const messageText = userInput.value.trim();
        if (!messageText) return;

        addMessage('user', messageText);
        userInput.value = ''; // Clear input field
        sendButton.disabled = true; // Disable button while waiting
        showTypingIndicator();
        productRecommendations.style.display = 'none'; // Hide products while waiting


        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: messageText }),
            });

            removeTypingIndicator(); // Remove indicator once response starts coming

            if (!response.ok) {
                // Try to get error message from response body
                let errorMsg = `Error: ${response.statusText}`;
                try {
                    const errorData = await response.json();
                    errorMsg = errorData.error || errorMsg;
                } catch (e) {
                    // Ignore if response body is not JSON
                }
                addMessage('assistant', `Sorry, something went wrong. ${errorMsg}`);
                console.error('API Error:', response.status, response.statusText);
            } else {
                const data = await response.json();
                if (data.response) {
                     addMessage('assistant', data.response);
                } else {
                     addMessage('assistant', 'Sorry, I received an empty response.'); // Handle case where text response is missing
                }
                displayProducts(data.products || []); // Display any returned products
            }

        } catch (error) {
            removeTypingIndicator();
            addMessage('assistant', 'Sorry, I couldn\'t connect to the server. Please check your connection and try again.');
            console.error('Network or Fetch Error:', error);
        } finally {
             sendButton.disabled = false; // Re-enable button
             userInput.focus(); // Set focus back to input
        }
    }

    // --- Event Listeners ---
    sendButton.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (event) => {
        // Send message on Enter key press
        if (event.key === 'Enter') {
            sendMessage();
        }
    });

     // Initial focus on input field
     userInput.focus();

});