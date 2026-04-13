import numpy as np


class GradientBasedSeedOptimizer:
    """智能种子优化器：用伪梯度引导种子搜索"""

    def __init__(self, base_seed=42, learning_rate=10.0):
        self.base_seed    = base_seed
        self.learning_rate = learning_rate
        self.seed_history  = []
        self.loss_history  = []
        self.best_seed     = base_seed
        self.best_loss     = float('inf')

    def compute_seed_gradient(self):
        if len(self.loss_history) < 2:
            return np.random.randn() * self.learning_rate
        recent_losses = self.loss_history[-3:]
        recent_seeds  = self.seed_history[-3:]
        if len(recent_losses) >= 2:
            loss_diff = recent_losses[-1] - recent_losses[-2]
            seed_diff = recent_seeds[-1] - recent_seeds[-2]
            gradient  = (loss_diff / seed_diff) if seed_diff != 0 else 0
            momentum  = np.random.uniform(-0.3, 0.3)
            noise     = np.random.randn() * 0.2
            return -gradient * self.learning_rate + momentum + noise
        return np.random.randn() * self.learning_rate

    def get_next_seed(self):
        if not self.seed_history:
            return self.base_seed
        gradient_step = self.compute_seed_gradient()
        if self.loss_history and self.loss_history[-1] < self.best_loss:
            next_seed = int(self.seed_history[-1] + gradient_step * 0.5)
        else:
            next_seed = int(self.best_seed + gradient_step)
        next_seed = max(1, abs(next_seed))
        if next_seed in self.seed_history:
            next_seed += np.random.randint(10, 100)
        return next_seed

    def update(self, seed, loss):
        self.seed_history.append(seed)
        self.loss_history.append(loss)
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_seed = seed

    def get_summary(self):
        return {
            'best_seed':    self.best_seed,
            'best_loss':    self.best_loss,
            'seed_history': self.seed_history.copy(),
            'loss_history': self.loss_history.copy(),
        }
