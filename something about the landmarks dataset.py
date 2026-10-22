import json
names = json.load(open("../checkpoints/class_names_landmarks.json"))
print(list(enumerate(names)))