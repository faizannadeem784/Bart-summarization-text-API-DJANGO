from transformers import pipeline, BartForConditionalGeneration, BartTokenizer

def generate_summary(input_text, model, tokenizer, max_length=500, min_length=50):
    """
    Generate a summary for the given input text using the specified model and tokenizer.

    Args:
        input_text (str): The input text for summarization.
        model: The pre-trained model for summarization.
        tokenizer: The pre-trained tokenizer for processing the input text.
        max_length (int): The maximum length of the generated summary.
        min_length (int): The minimum length of the generated summary.

    Returns:
        str: The generated summary text.
    """
    # Create a summarization pipeline
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
    
    # Generate summary using the pipeline
    summary = summarizer(input_text, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
    
    # Return the summary text
    return summary[0]['summary_text']

# Load a specific BART (Bidirectional and Auto-Regressive Transformers) model and tokenizer
model_name = "facebook/bart-large-cnn"
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)

# Example usage
input_paragraph = """
On Monday, Mistral AI announced a new AI language model called Mixtral 8x7B, a "mixture of experts" (MoE) 
model with open weights that reportedly truly matches OpenAI's GPT-3.5 in performance an achievement that 
has been claimed by others in the past but is being taken seriously by AI heavyweights such as OpenAI's 
Andrej Karpathy and Jim Fan. That means we're closer to having a ChatGPT-3.5-level AI assistant that can run 
freely and locally on our devices, given the right implementation.
"""

# Generate summary
output_summary = generate_summary(input_paragraph, model, tokenizer)

# Print original text and summary
print("Original Text:\n", input_paragraph)
print("\nSummary:\n", output_summary)