from manim import *

class BubbleSortVisualization(Scene):
    def construct(self):
        # Title
        title = Text("Bubble Sort Algorithm", font_size=40, color=YELLOW)
        title.to_edge(UP)
        self.play(Write(title))

        # Array to sort
        arr = [6, 2, 8, 4, 1]
        n = len(arr)

        # Create boxes and numbers
        boxes = VGroup(*[Square(side_length=1.2, color=BLUE) for _ in arr])
        boxes.arrange(RIGHT, buff=0.2).shift(DOWN)

        numbers = VGroup(*[Text(str(num), font_size=36) for num in arr])
        for i in range(n):
            numbers[i].move_to(boxes[i].get_center())

        # Display the array
        self.play(FadeIn(boxes), FadeIn(numbers))
        self.wait(1)

        # Bubble sort animation
        for i in range(n - 1):
            for j in range(n - i - 1):
                # Highlight compared boxes
                self.play(
                    boxes[j].animate.set_color(YELLOW),
                    boxes[j+1].animate.set_color(YELLOW),
                    run_time=0.4
                )
                self.wait(0.2)

                # Compare numbers
                if arr[j] > arr[j + 1]:
                    # Swap
                    self.play(
                        numbers[j].animate.move_to(boxes[j + 1].get_center()),
                        numbers[j + 1].animate.move_to(boxes[j].get_center()),
                        run_time=0.6
                    )
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
                    numbers[j], numbers[j + 1] = numbers[j + 1], numbers[j]

                # Reset colors
                self.play(
                    boxes[j].animate.set_color(BLUE),
                    boxes[j + 1].animate.set_color(BLUE),
                    run_time=0.3
                )

            # Mark last element as sorted
            boxes[n - i - 1].set_color(GREEN)
            self.play(Indicate(boxes[n - i - 1]))

        # Mark first element as sorted
        boxes[0].set_color(GREEN)
        self.play(Indicate(boxes[0]))

        # Final sorted display
        sorted_text = Text("Sorted Array!", font_size=36, color=GREEN)
        sorted_text.next_to(boxes, DOWN * 2)
        self.play(Write(sorted_text))
        self.wait(2)

        # End
        self.play(FadeOut(boxes), FadeOut(numbers), FadeOut(sorted_text), FadeOut(title))

