"""
Improved benchmark tracker for Autonomous Cloud Security Orchestrator.
Ensures model is properly evaluated across federated rounds.
"""
import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BenchmarkTracker:
    def __init__(self, model_dir="server", 
                 results_dir="benchmark_results",
                 experiment_name="baseline",
                 num_test_samples=1000,
                 feature_count=10,
                 threat_ratio=0.2,
                 random_seed=42):
        """Initialize the benchmark tracker."""
        self.model_dir = model_dir
        self.results_dir = results_dir
        self.experiment_name = experiment_name
        self.num_test_samples = num_test_samples
        self.feature_count = feature_count
        self.threat_ratio = threat_ratio
        self.random_seed = random_seed
        
        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize results storage
        self.results = {
            'round': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'true_positive_rate': [],
            'false_positive_rate': [],
            'evaluation_time': [],
            'timestamp': [],
            'threat_prob_mean': [] # Average probability of threat class
        }
        
        # Generate test data once for consistent evaluation
        self.X_test, self.y_test = self._generate_test_data()
        logger.info(f"Generated test dataset with {self.num_test_samples} samples "
                   f"({int(self.num_test_samples * self.threat_ratio)} threats, "
                   f"{int(self.num_test_samples * (1-self.threat_ratio))} normal)")
        
    def _generate_test_data(self):
        """Generate synthetic test data for evaluation with clear separation."""
        # Set seed for reproducibility
        np.random.seed(self.random_seed)
        
        # Calculate number of samples for each class
        num_threats = int(self.num_test_samples * self.threat_ratio)
        num_normal = self.num_test_samples - num_threats
        
        # Generate normal data (lower feature values)
        normal_data = np.random.rand(num_normal, self.feature_count) * 0.3
        
        # Generate threat data with stronger signal (higher values for specific features)
        threat_data = np.random.rand(num_threats, self.feature_count) * 0.3
        # Make last 3 features much higher for threats
        threat_data[:, -3:] = 0.7 + np.random.rand(num_threats, 3) * 0.3
        
        # Print sample data to verify separation
        logger.info(f"Normal data sample (first row): {normal_data[0]}")
        logger.info(f"Threat data sample (first row): {threat_data[0]}")
        
        # Combine data
        X = np.vstack((normal_data, threat_data))
        y = np.concatenate((np.zeros(num_normal, dtype=int), np.ones(num_threats, dtype=int)))
        
        # Shuffle data
        indices = np.arange(self.num_test_samples)
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
        
        return X, y
    
    def evaluate_model(self, round_num, model_file=None):
        """Evaluate model performance for a specific round."""
        if model_file is None:
            model_file = os.path.join(self.model_dir, "global_model.pkl")
        
        if not os.path.exists(model_file):
            logger.error(f"Model file not found: {model_file}")
            return None
        
        try:
            # Load model
            start_time = time.time()
            model = joblib.load(model_file)
            
            # Print model coefficients for debugging
            logger.info(f"Model coefficients: {model.coef_}")
            logger.info(f"Model intercept: {model.intercept_}")
            
            # Evaluate model
            y_pred = model.predict(self.X_test)
            y_prob = model.predict_proba(self.X_test)[:, 1]  # Probability of class 1 (threat)
            
            # Log class distribution in predictions
            unique, counts = np.unique(y_pred, return_counts=True)
            pred_dist = dict(zip(unique, counts))
            logger.info(f"Prediction distribution: {pred_dist}")
            logger.info(f"Mean threat probability: {np.mean(y_prob):.4f}")
            
            # Calculate metrics
            acc = accuracy_score(self.y_test, y_pred)
            prec = precision_score(self.y_test, y_pred, zero_division=0)
            rec = recall_score(self.y_test, y_pred, zero_division=0)
            f1 = f1_score(self.y_test, y_pred, zero_division=0)
            
            # Calculate ROC metrics
            try:
                tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).ravel()
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # True positive rate (sensitivity)
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False positive rate
            except ValueError:
                # This happens if model predicts only one class
                logger.warning("Could not calculate confusion matrix - model may predict only one class")
                if all(y_pred == 0):  # All predictions are negative
                    tp, fp = 0, 0
                    tn = sum(self.y_test == 0)
                    fn = sum(self.y_test == 1)
                elif all(y_pred == 1):  # All predictions are positive
                    tn, fn = 0, 0
                    tp = sum(self.y_test == 1)
                    fp = sum(self.y_test == 0)
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            # Calculate evaluation time
            eval_time = time.time() - start_time
            
            # Store results
            self.results['round'].append(round_num)
            self.results['accuracy'].append(float(acc))
            self.results['precision'].append(float(prec))
            self.results['recall'].append(float(rec))
            self.results['f1'].append(float(f1))
            self.results['true_positive_rate'].append(float(tpr))
            self.results['false_positive_rate'].append(float(fpr))
            self.results['evaluation_time'].append(float(eval_time))
            self.results['timestamp'].append(time.time())
            self.results['threat_prob_mean'].append(float(np.mean(y_prob)))
            
            logger.info(f"Round {round_num} Metrics: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")
            logger.info(f"Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
            
            return {
                'round': round_num,
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'true_positive_rate': tpr,
                'false_positive_rate': fpr,
                'evaluation_time': eval_time,
                'threat_prob_mean': float(np.mean(y_prob))
            }
            
        except Exception as e:
            logger.error(f"Error evaluating model for round {round_num}: {e}", exc_info=True)
            return None
    
    def save_results(self):
        """Save results to CSV and JSON files."""
        # Create DataFrame from results
        df = pd.DataFrame(self.results)
        
        # Save to CSV
        csv_file = os.path.join(self.results_dir, f"{self.experiment_name}_metrics.csv")
        df.to_csv(csv_file, index=False)
        
        # Save to JSON
        json_file = os.path.join(self.results_dir, f"{self.experiment_name}_metrics.json")
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=4)
        
        logger.info(f"Results saved to {csv_file} and {json_file}")
        
        return csv_file, json_file
    
    def plot_convergence(self, metric='f1', save_fig=True):
        """Plot convergence of specified metric over rounds."""
        if metric not in self.results:
            logger.error(f"Metric {metric} not found in results")
            return None
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.results['round'], self.results[metric], marker='o', linewidth=2)
        plt.title(f"{metric.capitalize()} Convergence ({self.experiment_name})")
        plt.xlabel("Federated Round")
        plt.ylabel(metric.capitalize())
        plt.grid(True)
        
        # Add value labels at each point
        for x, y in zip(self.results['round'], self.results[metric]):
            plt.annotate(f"{y:.3f}", (x, y), textcoords="offset points", 
                         xytext=(0,10), ha='center')
        
        if save_fig:
            fig_file = os.path.join(self.results_dir, f"{self.experiment_name}_{metric}_convergence.png")
            plt.savefig(fig_file)
            logger.info(f"Convergence plot saved to {fig_file}")
            plt.close()
            return fig_file
        else:
            plt.show()
            return None

    def plot_multi_metric(self, metrics=['accuracy', 'precision', 'recall', 'f1'], save_fig=True):
        """Plot multiple metrics together for comparison."""
        plt.figure(figsize=(12, 7))
        
        for metric in metrics:
            if metric in self.results:
                plt.plot(self.results['round'], self.results[metric], 
                         marker='o', linewidth=2, label=metric.capitalize())
        
        plt.title(f"Learning Metrics Comparison ({self.experiment_name})")
        plt.xlabel("Federated Round")
        plt.ylabel("Metric Value")
        plt.grid(True)
        plt.legend()
        
        if save_fig:
            fig_file = os.path.join(self.results_dir, f"{self.experiment_name}_multi_metric.png")
            plt.savefig(fig_file)
            logger.info(f"Multi-metric plot saved to {fig_file}")
            plt.close()
            return fig_file
        else:
            plt.show()
            return None


def main():
    """Example usage of BenchmarkTracker."""
    tracker = BenchmarkTracker(experiment_name="baseline_test")
    
    # Evaluate model for each round
    for round_num in range(5):  # Assuming 5 rounds
        # You'd normally wait for each round to complete
        # For standalone testing, we just evaluate the same model multiple times
        tracker.evaluate_model(round_num)
    
    # Save results
    tracker.save_results()
    
    # Plot convergence
    tracker.plot_convergence(metric='f1')
    tracker.plot_convergence(metric='accuracy')
    tracker.plot_multi_metric()


if __name__ == "__main__":
    main()
