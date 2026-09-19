import math
import random
from pathlib import Path


def linear_regression(xs, ys):
    if len(xs) != len(ys) or not xs:
        raise ValueError("xs and ys must be non-empty and same length")

    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)

    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator == 0:
        raise ValueError("All x values are identical; slope is undefined")

    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    return slope, intercept


def predict_linear(x, slope, intercept):
    return slope * x + intercept


def plot_linear_regression(xs, ys, slope, intercept, output_path="linear_regression_plot.svg"):
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 4))
        plt.scatter(xs, ys, color="blue", label="Data points")

        x_min, x_max = min(xs), max(xs)
        y_min = predict_linear(x_min, slope, intercept)
        y_max = predict_linear(x_max, slope, intercept)
        plt.plot([x_min, x_max], [y_min, y_max], color="red", label="Best fit line")

        plt.title("Basic Linear Regression")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        return
    except ModuleNotFoundError:
        pass

    width, height = 640, 420
    margin = 40
    x_min, x_max = min(xs), max(xs)
    y_line = [predict_linear(x, slope, intercept) for x in (x_min, x_max)]
    y_all = ys + y_line
    y_min, y_max = min(y_all), max(y_all)

    def map_x(x):
        scale = (x - x_min) / (x_max - x_min) if x_max != x_min else 0.5
        return margin + scale * (width - 2 * margin)

    def map_y(y):
        scale = (y - y_min) / (y_max - y_min) if y_max != y_min else 0.5
        return height - margin - scale * (height - 2 * margin)

    points = "\n".join(
        f'<circle cx="{map_x(x):.2f}" cy="{map_y(y):.2f}" r="4" fill="blue" />'
        for x, y in zip(xs, ys)
    )

    line = (
        f'<line x1="{map_x(x_min):.2f}" y1="{map_y(y_line[0]):.2f}" '
        f'x2="{map_x(x_max):.2f}" y2="{map_y(y_line[1]):.2f}" '
        'stroke="red" stroke-width="2" />'
    )

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect x="0" y="0" width="{width}" height="{height}" fill="white" />
<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="black" />
<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="black" />
{line}
{points}
</svg>
'''
    Path(output_path).write_text(svg, encoding="utf-8")


def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    z = math.exp(x)
    return z / (1 + z)


def train_xor(epochs=12000, learning_rate=0.7, seed=42):
    random.seed(seed)

    X = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
    y = [0.0, 1.0, 1.0, 0.0]

    w1 = [[random.uniform(-1, 1), random.uniform(-1, 1)] for _ in range(2)]
    b1 = [random.uniform(-1, 1), random.uniform(-1, 1)]
    w2 = [random.uniform(-1, 1), random.uniform(-1, 1)]
    b2 = random.uniform(-1, 1)

    for _ in range(epochs):
        for (x1, x2), target in zip(X, y):
            h_in_0 = x1 * w1[0][0] + x2 * w1[0][1] + b1[0]
            h_in_1 = x1 * w1[1][0] + x2 * w1[1][1] + b1[1]
            h_out_0 = sigmoid(h_in_0)
            h_out_1 = sigmoid(h_in_1)

            out_in = h_out_0 * w2[0] + h_out_1 * w2[1] + b2
            out = sigmoid(out_in)

            out_error = out - target
            out_delta = out_error * out * (1 - out)

            h_delta_0 = out_delta * w2[0] * h_out_0 * (1 - h_out_0)
            h_delta_1 = out_delta * w2[1] * h_out_1 * (1 - h_out_1)

            w2[0] -= learning_rate * out_delta * h_out_0
            w2[1] -= learning_rate * out_delta * h_out_1
            b2 -= learning_rate * out_delta

            w1[0][0] -= learning_rate * h_delta_0 * x1
            w1[0][1] -= learning_rate * h_delta_0 * x2
            w1[1][0] -= learning_rate * h_delta_1 * x1
            w1[1][1] -= learning_rate * h_delta_1 * x2
            b1[0] -= learning_rate * h_delta_0
            b1[1] -= learning_rate * h_delta_1

    def predict(a, b):
        h0 = sigmoid(a * w1[0][0] + b * w1[0][1] + b1[0])
        h1 = sigmoid(a * w1[1][0] + b * w1[1][1] + b1[1])
        o = sigmoid(h0 * w2[0] + h1 * w2[1] + b2)
        return o

    predictions = [predict(a, b) for a, b in X]

    model = {"w1": w1, "b1": b1, "w2": w2, "b2": b2}
    return model, predictions


def run_demo(output_dir="."):
    xs = [1, 2, 3, 4, 5]
    ys = [2, 4, 5, 4, 5]

    slope, intercept = linear_regression(xs, ys)
    output_path = Path(output_dir) / "linear_regression_plot.svg"
    plot_linear_regression(xs, ys, slope, intercept, str(output_path))

    _, xor_predictions = train_xor()

    return {
        "linear_regression": {
            "slope": slope,
            "intercept": intercept,
            "plot": str(output_path),
        },
        "xor_predictions": xor_predictions,
    }


if __name__ == "__main__":
    result = run_demo()
    print("Linear Regression:")
    print(f"  slope={result['linear_regression']['slope']:.4f}")
    print(f"  intercept={result['linear_regression']['intercept']:.4f}")
    print(f"  graph={result['linear_regression']['plot']}")

    print("XOR Predictions:")
    for inp, pred in zip([(0, 0), (0, 1), (1, 0), (1, 1)], result["xor_predictions"]):
        print(f"  {inp} -> {pred:.4f}")
