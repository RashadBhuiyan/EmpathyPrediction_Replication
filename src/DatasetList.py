# Empathy from Perspectives (EFP) Dataset class
# Headers: [,place_A,why_A,story_A,place_B,why_B,story_B,empathy]
# empathy scores are normalized from 0.0 to 1.0 (divide by 100)
# Empathic Stories (ES) Dataset class
# Headers: [,pairs,binned,story_A,story_B,story_A_summary,story_B_summary,Empathic Similarity (gpt3.5),Empathic Similarity Binned (gpt3.5),Empathic Similarity Reasons (gpt3.5),similarity_empathy_human_AGG,similarity_event_human_AGG,similarity_emotion_human_AGG,similarity_moral_human_AGG]
# similarity_empathy_human_AGG scores are normalized from 0.0 to 1.0 (divide by 4.0)
# Concatenated Dataset class for EFP and ES

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
    
class ContextDataset(torch.utils.data.Dataset):
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

        # combine relevant contexts
        context_A = place_A + "[SEP]" + why_A + "[SEP]" + story_A
        context_B = place_B + "[SEP]" + why_B + "[SEP]" + story_B
        empathy_score = i["empathy"] / 100.0

        return [context_A, context_B, empathy_score]

    def __len__(self):
        if self.limit!=-1:
            return self.limit
        return len(self.data)

class ESDataset(torch.utils.data.Dataset):
    def __init__(self, data, limit=-1):
        self.data = data
        self.limit = limit

    def __getitem__(self, idx):
        i = self.data.iloc[idx]
        place_A = ""
        place_B = ""
        why_A = ""
        why_B = ""
        story_A = i["story_A"]
        story_B = i["story_B"]
        empathy_score = i["similarity_empathy_human_AGG"] / 4.0

        return [place_A, why_A, story_A, place_B, why_B, story_B, empathy_score]

    def __len__(self):
        if self.limit!=-1:
            return self.limit
        return len(self.data)

class ConcatenatedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __getitem__(self, idx):
        if idx < len(self.dataset1):
            return self.dataset1[idx]
        else:
            return self.dataset2[idx - len(self.dataset1)]

    def __len__(self):
        return len(self.dataset1) + len(self.dataset2)