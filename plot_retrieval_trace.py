import matplotlib.pyplot as plt
import pandas as pd


data = pd.DataFrame({
    "Rank": [1, 2, 3, 4, 5],
    "Chunk": [
        "aoa_CHUNK_0091",
        "aoa_CHUNK_0077",
        "aoa_CHUNK_0090",
        "aoa_CHUNK_0092",
        "aoa_CHUNK_0122",
    ],
    "Dense Score": [
        0.659,
        0.520,
        0.528,
        0.450,
        0.563,
    ],
    "Cross-Encoder Score": [
        7.925,
        3.650,
        2.119,
        1.731,
        -0.353,
    ],
})

if __name__ == "__main__":
    labels = [
        f"#{rank}\n{chunk.replace('aoa_CHUNK_', '')}"
        for rank, chunk in zip(data["Rank"], data["Chunk"])
    ]

    colors = [
        "#4C78A8",
        "#72A3D0",
        "#72A3D0",
        "#72A3D0",
        "#A8A8A8",
    ]

    fig, axis = plt.subplots(figsize=(9, 5))

    bars = axis.bar(
        labels,
        data["Cross-Encoder Score"],
        color=colors,
    )

    axis.axhline(
        y=0,
        color="black",
        linewidth=0.8,
    )

    axis.set_xlabel("Final Rank and Chunk")
    axis.set_ylabel("Cross-Encoder Score")
    axis.set_title("Cross-Encoder Reranking Results")
    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.5,
    )

    for bar, score in zip(bars, data["Cross-Encoder Score"]):
        height = bar.get_height()

        if height >= 0:
            position = height + 0.25
            vertical_alignment = "bottom"
        else:
            position = height - 0.55
            vertical_alignment = "top"

        axis.text(
            bar.get_x() + bar.get_width() / 2,
            position,
            f"{score:.3f}",
            ha="center",
            va=vertical_alignment,
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(
        "retrieval_trace_reranker.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()