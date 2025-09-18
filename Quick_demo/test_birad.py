import csv
import tqdm.auto as tqdm
import torch
from torchvision import transforms
from PIL import Image
from transformers import LlamaTokenizer
from Model.RadFM.multimodality_model import MultiLLaMAForCausalLM

# ----------------------------
# Setup Tokenizer with image tokens
# ----------------------------
def get_tokenizer(tokenizer_path, max_img_size=100, image_num=32):
    if isinstance(tokenizer_path, str):
        text_tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
        special_token = {"additional_special_tokens": ["<image>", "</image>"]}
        image_padding_tokens = []

        for i in range(max_img_size):
            image_padding_token = ""
            for j in range(image_num):
                image_token = f"<image{i*image_num+j}>"
                image_padding_token += image_token
                special_token["additional_special_tokens"].append(image_token)

            image_padding_tokens.append(image_padding_token)
            text_tokenizer.add_special_tokens(special_token)

        text_tokenizer.pad_token_id = 0
        text_tokenizer.bos_token_id = 1
        text_tokenizer.eos_token_id = 2

    return text_tokenizer, image_padding_tokens


# ----------------------------
# Combine text + image into model input
# ----------------------------
def combine_and_preprocess(question, img_path, image_padding_tokens):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    image = Image.open(img_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0).unsqueeze(-1)  # (c,h,w,1)

    target_H, target_W, target_D = 512, 512, 4
    vision_x = torch.nn.functional.interpolate(image, size=(target_H, target_W, target_D))
    vision_x = vision_x.unsqueeze(0)  # add batch

    # Insert placeholder token
    text = "<image>" + image_padding_tokens[0] + "</image>" + question
    return text, vision_x


# ----------------------------
# Main
# ----------------------------
def main():
    print("Setup tokenizer")
    tokenizer, image_padding_tokens = get_tokenizer("./Language_files")

    print("Setup model")
    model = MultiLLaMAForCausalLM(lang_model_path="./Language_files")
    ckpt = torch.load("./pytorch_model.bin", map_location="cpu")
    model.load_state_dict(ckpt, strict=False)
    model = model.to("cuda").eval()

    # Fixed prompt
    question = "What is the BIRADS score for this mammograpy image?"

    # Load image list from CSV
    image_csv = "../src/dmid_test.csv"
    with open(image_csv, "r") as f:
        reader = csv.DictReader(f)
        image_paths = [row["image_path"] for row in reader]

    # Output results
    print("Start testing")
    with open("output_dmid_birad.csv", "w", newline="") as outf:
        writer = csv.writer(outf)
        writer.writerow(["image_path", "prompt", "prediction"])

        for img_path in tqdm.tqdm(image_paths):
            try:
                print("testing image:", img_path)
                text, vision_x = combine_and_preprocess(question, img_path, image_padding_tokens)
                lang_x = tokenizer(text, max_length=2048, truncation=True, return_tensors="pt")["input_ids"].to("cuda")
                vision_x = vision_x.to("cuda")

                with torch.no_grad():
                    generation = model.generate(lang_x, vision_x)
                    pred = tokenizer.batch_decode(generation, skip_special_tokens=True)[0]

                writer.writerow([img_path, question, pred])
            except Exception as e:
                writer.writerow([img_path, question, f"[ERROR: {e}]"])
            
            print("Testing completed for image:", img_path)

    print("Saved predictions to output_dmid_demo_birad.csv")


if __name__ == "__main__":
    main()
