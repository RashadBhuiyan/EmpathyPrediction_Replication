# Empathy from Perspectives (EFP) Dataset class
# Headers: [,place_A,why_A,story_A,place_B,why_B,story_B,empathy]
# empathy scores are normalized from 0.0 to 1.0 (divide by 100)

import torch

class EFPDataset(torch.utils.data.Dataset):
    def __init__(self, data, limit=-1):
        self.data = data
        self.limit = limit

    def __getitem__(self, idx):
        i = self.data.iloc[idx]
        place_A = i["place_A"]
        place_B = i["place_B"]
        why_A = i["why_A"]
        why_B = i["why_B"]
        story_A = i["story_A"]
        story_B = i["story_B"]
        empathy_score = i["empathy"] / 100.0

        return [place_A, why_A, story_A, place_B, why_B, story_B, empathy_score]

    def __len__(self):
        if self.limit!=-1:
            return self.limit
        return len(self.data)