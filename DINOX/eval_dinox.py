import os
import json
from datasets import load_dataset
from gdino import GroundingDINO
import cv2
from tqdm import tqdm

class ObjectCountEvaluator:
    def __init__(self):
        self.gdino = GroundingDINO()
        self.dataset = load_dataset("nyu-visionx/VSI-Bench")
        
        # 设置数据集根目录
        self.data_root = "data/VSI-Bench"  # 你需要把数据集下载到这个目录
        os.makedirs(self.data_root, exist_ok=True)
    def get_image_path(self, dataset, scene_name):
        """构建图片路径"""
        # 根据数据集类型构建路径
        if dataset == "arkitscenes":
            return os.path.join(self.data_root, "arkitscenes", scene_name, "frames", "frame_0.jpg")
        elif dataset == "scannet":
            return os.path.join(self.data_root, "scannet", scene_name, "frame_0.jpg")
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

    def evaluate_single_frame(self, image_path, question):
        """Evaluate a single frame for object counting"""
        if not os.path.exists(image_path):
            print(f"Warning: Image not found at {image_path}")
            return None, None
            
        # Extract object type from question
        # Example: "How many table(s) are in this room?"
        object_type = question.lower().split("how many ")[1].split("(s)")[0].strip()
        
        # Get DINO-X predictions
        predictions = self.gdino.get_dinox(image_path, [object_type])
        
        # Count objects of the specified type
        count = sum(1 for pred in predictions if pred.category.lower() == object_type)
        
        return count, predictions

    def evaluate_object_count(self, output_dir="results"):
        """Evaluate all object counting questions in VSI-Bench"""
        os.makedirs(output_dir, exist_ok=True)
        results = []
        
        # Filter for object count questions
        count_questions = [
            sample for sample in self.dataset["test"] 
            if sample["question_type"] == "object_counting"
        ]
        
        print(f"Found {len(count_questions)} object counting questions")
        
        for sample in tqdm(count_questions[:10]):
            image_path = self.get_image_path(sample["dataset"], sample["scene_name"])
            question = sample["question"]
            ground_truth = int(sample["ground_truth"])
            
            predicted_count, predictions = self.evaluate_single_frame(image_path, question)
            print(f"predicted_count: {predicted_count}, ground_truth: {ground_truth}")
            if predicted_count is None:
                continue
                
            result = {
                "id": sample["id"],
                "scene_name": sample["scene_name"],
                "dataset": sample["dataset"],
                "question": question,
                "predicted_count": predicted_count,
                "ground_truth": ground_truth,
                "correct": predicted_count == ground_truth,
                "image_path": image_path
            }
            results.append(result)
            
            # Optionally visualize detection
            if predictions:
                self.gdino.visualize_bbox_and_mask(
                    predictions=predictions,
                    img_path=image_path,
                    output_dir=os.path.join(output_dir, "dinox"),
                    output_name=os.path.basename(image_path)
                )
        
        if not results:
            print("No valid results found. Please check if images are downloaded correctly.")
            return None
            
        # Calculate accuracy
        accuracy = sum(r["correct"] for r in results) / len(results)
        
        # Save results
        with open(os.path.join(output_dir, "object_count_results.json"), "w") as f:
            json.dump({
                "results": results,
                "accuracy": accuracy,
                "total_samples": len(results)
            }, f, indent=2)
        
        print(f"Object counting accuracy: {accuracy:.2%}")
        return accuracy

if __name__ == "__main__":
    evaluator = ObjectCountEvaluator()
    evaluator.evaluate_object_count()