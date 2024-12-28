from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def test_model_inference(model_path):
    try:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            use_safetensors=True,
            torch_dtype=torch.float32
        )

        print("\\nTesting basic inference...")
        test_input = "The quick brown fox"
        inputs = tokenizer(test_input, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=10,
                num_return_sequences=1
            )

        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Input: {test_input}")
        print(f"Output: {result}")
        return True

    except Exception as e:
        print(f"Error during test: {str(e)}")
        return False

# 사용 예
success = test_model_inference("./")
if success:
    print("\\nModel basic functionality test passed!")
else:
    print("\\nModel test failed!")