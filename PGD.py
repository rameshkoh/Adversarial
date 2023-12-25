import torch
import torch.nn.functional as F

def pgd_attack(model, input_image, label, epsilon, alpha, num_iter):
    """Performs the PGD attack on an input image."""
    perturbed_image = input_image.clone()

    for i in range(num_iter):
        perturbed_image.requires_grad = True
        output = model(perturbed_image)

        model.zero_grad()
        loss = F.cross_entropy(output, label)
        loss.backward()

        # Update the image with a small step in the direction of the gradient
        data_grad = perturbed_image.grad.data
        step = alpha * data_grad.sign()
        perturbed_image = perturbed_image + step

        # Project the perturbed image back into the epsilon-ball around the original image
        perturbation = torch.clamp(perturbed_image - input_image, min=-epsilon, max=epsilon)
        perturbed_image = torch.clamp(input_image + perturbation, 0, 1).detach_()

    return perturbed_image

# Example usage
model = ... # Assume a pre-loaded and pre-trained model
input_image = ... # Pre-processed input image
label = ... # Correct label for the input image
epsilon = 0.03
alpha = 0.01
num_iter = 40

adversarial_image = pgd_attack(model, input_image, label, epsilon, alpha, num_iter)
