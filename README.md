# Customer_support_chatbot

![Screenshot 2025-02-26 114540](https://github.com/user-attachments/assets/90d07be5-9c35-4fed-a3c1-dbe276817b4f)

## Overview
Customer support plays a crucial role in modern business operations, ensuring user satisfaction and operational efficiency in industries ranging from e-commerce to telecommunications. As customer inquiries grow in volume and complexity, traditional manual response systems become increasingly impractical, leading to long wait times and inconsistencies in support quality. To address this, businesses are turning to AI-powered chatbots that can efficiently handle routine queries.

This repository contains a domain-specific customer support chatbot developed using T5-small, fine-tuned on the full 26,872-row Bitext Customer Support Dataset. The chatbot automates routine customer inquiries, such as order cancellations, refund tracking, and payment issues, delivering natural, coherent responses via an intuitive Gradio interface. It’s optimized for Google Colab’s free tier, demonstrating efficient customer support automation within resource constraints.


## Dataset
- **Source**: Bitext Customer Support Dataset (available on Hugging Face at `bitext/Bitext-customer-support-llm-chatbot-training-dataset`).
- **Size**: 26,872 structured conversational pairs, covering 27 intents (e.g., `cancel_order`, `track_refund`, `payment_issue`).
- **Structure**: Each pair includes a customer inquiry (e.g., “I want to cancel my order”), intent (e.g., “cancel_order”), and response (e.g., “To cancel your order, log in to our website…”). I verified no missing values and balanced intents (880-913 samples per intent, visualized in `intent_distribution.png`).
  ![image](https://github.com/user-attachments/assets/605fe9cf-5b4c-4609-a16d-d47d207f3fb4)

- **Preprocessing**: I normalized text (lowercasing, removing special chars), tokenized with T5-small’s WordPiece (`max_length=128`), and split 90% for training (24,184 samples) and 10% for validation (2,688 samples).

## Performance Metrics
- **Perplexity**: ~2.61, calculated from the tuned evaluation loss of 0.96, reflecting the accuracy for T5-small’s generative responses.
- **Qualitative Evaluation**: Tested in Streamlit with sample queries, showing:
  - In-domain accuracy (e.g., “cancel my order” → “To cancel your order, log in to our website and navigate to the ‘Orders’ section”).
  - Out-of-domain rejection (e.g., “what’s the weather?” → “I’m sorry, but I didn’t understand that. Can you please rephrase your question or ask about something else?”).
- **Training**: Fine-tuned over 3 epochs, comparing learning rates 2e-5 (loss 1.26) and 5e-5 (loss 0.96), achieving a ~24% loss reduction for optimal performance.

- **Experiment table**
  
| Dataset Size    | Epochs | Learning Rate (Baseline) | Loss (Baseline) | Learning Rate (Tuned) | Loss (Tuned) | Notes                                      |  
|-----------------|--------|--------------------------|-----------------|-----------------------|--------------|--------------------------------------------|  
| 5,000 Subset    | 3      | 2e-5                     | 1.82            | 5e-5                  | 1.32         | Initial test, limited data, moderate performance |  
| 5,000 Subset    | 10     | 2e-5                     | 1.37            | 5e-5                  | 1.04         | Improved with more epochs, but overfit risk |  
| Full (26,872)   | 3      | 2e-5                     | 1.26            | 5e-5                  | 0.96         | Final choice, best balance of performance and efficiency |

## Steps to Run the Chatbot
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/customer-support-chatbot.git
   cd customer-support-chatbot

# Customer Support Chatbot

2. Install Dependencies
Install required Python packages using:

```bash
pip install -r requirements.txt
```

3. Set Up Environment
- Ensure you have **Python 3.9+** and **Git** installed.  
- Optionally, use **Google Colab** for GPU support (recommended for T5-small training).  

4.  Run the Chatbot
### **Local Execution**
Run the notebook in **Jupyter Notebook** or **Colab** for training and evaluation.  

### **Streamlit Deployment**

Access the **Streamlit** interface via the public URL ([https://customersupportchbot.streamlit.app/](https://customersupportchbot.streamlit.app/)).  


## Examples of Conversations
Here are sample interactions with the chatbot:

### **Example 1: Order Cancellation**
**User:** “cancel my order”  
**Chatbot:** “I'm here to assist you in canceling your order. To cancel your order, please follow these steps: 1. Log in to your account on our website. 2. Navigate to the "my orders" or "account" section. 3. Look for the option to cancel your order. 4. Click on the order to proceed with the cancellation request. 5. Select the order that you would like to cancel and click on it. If you encounter any difficulties or have further questions, feel free to let me know.”  

### **Example 2: Refund Tracking**
**User:** “track refund status”  
**Chatbot:** “To track your refund, provide your refund ID or check the ‘Refunds’ section on our website.”  

### **Example 3: Payment Issue**
**User:** “payment issue”  
**Chatbot:** “I'm here to assist you in resolving the payment issue. Your feedback is invaluable to us, and we appreciate your patience as we work towards a resolution. To better assist you, could you please provide me with more details about the specific payment issue you encountered? this will enable us to investigate the issue and provide you with the most accurate guidance. Thank you for your cooperation and cooperation.”  

### **Example 4: Password Reset**
**User:** “help me reset my password”  
**Chatbot:** “To reset your password, go to our website’s ‘Forgot Password’ section and follow the prompts.”  

### **Example 5: Out-of-Domain Query**
**User:** “what’s the weather?”  
**Chatbot:** “I’m sorry, but I didn’t understand that. Can you please rephrase your question or ask about something else?”  

---

## Files
- `Summative_Customer_support_chatbot.ipynb`: Colab notebook with code for data preprocessing, model training, evaluation, and Gradio deployment.  
- `intent_distribution.png`: Visualization of intent distribution in the training set.  
- `loss_comparison.png`: Bar chart comparing training losses (2e-5 vs. 5e-5).  
- `tokenized_dataset`: Preprocessed Bitext dataset for training.  
- `final_model`: Trained T5-small model and tokenizer.  
- `requirements.txt`: Python dependencies for the project.  

