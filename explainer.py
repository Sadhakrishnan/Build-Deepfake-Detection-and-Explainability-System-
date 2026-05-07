import os
import openai

def generate_explanation(fake_probability, suspicious_regions, metadata=None):
    """
    Uses OpenAI GPT to generate a forensic explanation based on model outputs.
    """
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    # Check if API key is present
    if not openai.api_key:
        return f"Explanation generation disabled (missing OPENAI_API_KEY). The model predicts this media as fake with {fake_probability * 100:.2f}% confidence."

    prompt = f"""
    You are an AI forensics expert analyzing a piece of media for deepfake manipulation.
    
    The model predicts this media as fake with {fake_probability * 100:.2f}% confidence.
    
    Detected suspicious regions from Grad-CAM analysis:
    {', '.join(suspicious_regions) if suspicious_regions else 'None explicitly identified'}
    
    Generate a professional, forensic-style explanation for why this media may be manipulated. 
    Keep it concise (1-2 sentences) and objective.
    """
    
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a professional digital forensics analyst."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating explanation: {e}")
        return f"The media likely contains deepfake manipulation based on a confidence score of {fake_probability * 100:.2f}%. Forensics explanation unavailable due to API error."
