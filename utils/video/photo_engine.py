import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

class PhotoEngine:
    #model_name: "default" gives the default engine
    def __init__(self, model_name: str):
        if (model_name == "default"):
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
            self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
        else:
            self.processor = BlipProcessor.from_pretrained(model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(model_name)
    
    def open_image (self, img_url:str):
        return Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

    def describe_image(self, opened_image) -> str:
        inputs = self.processor(opened_image, return_tensors="pt")
        out = self.model.generate(**inputs)
        return self.processor.decode(out[0], skip_special_tokens=True)

# Example Usage

engine = VideoEngine("default")
opened_image = engine.open_image("https://z.rajan.sh/robot.jpg")
print(engine.describe_image(opened_image))