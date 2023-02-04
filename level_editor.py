import json
from enum import Enum
from typing import Literal

import pygame

IncrementValues = Literal[0] | Literal[1] | Literal[-1]

FPS = 60


class CellState(Enum):
    EMPTY = (70, 70, 70)
    WALL = (0, 0, 0)
    PLAYER = (0, 255, 0)
    ENEMY = (255, 0, 0)


class LevelEditor:
    def __init__(self, screen_width: int = 800, screen_height: int = 600):
        self.screen_width = screen_width
        self.screen_height = screen_height

        self.grid_margin_x = 50
        self.grid_margin_y = 50

        self.grid_width = 25
        self.grid_height = 25

        self.cell_size = min(
            (screen_width - (self.grid_margin_x * 2)) // self.grid_width,
            (screen_height - (self.grid_margin_y * 2)) // self.grid_height,
        )

        self.grid_surface = pygame.Surface(
            (self.cell_size * self.grid_width, self.cell_size * self.grid_height)
        )
        self.grid_rect = self.grid_surface.get_rect(
            center=(self.screen_width // 2, self.screen_height // 2)
        )

        self.grid = [
            [CellState.EMPTY for _ in range(self.grid_width)]
            for _ in range(self.grid_height)
        ]

        self.left_click_draw_mode: Literal[CellState.EMPTY] | Literal[
            CellState.WALL
        ] = CellState.WALL

        self.window = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Level Editor")

    def update_grid_dimensions(
        self,
        width_increment: IncrementValues = 0,
        height_increment: IncrementValues = 0,
    ):
        if width_increment == -1 and self.grid_width > 1:
            for row in self.grid:
                row.pop()

        elif width_increment == 1:
            for row in self.grid:
                row.append(CellState.EMPTY)

        if height_increment == -1 and self.grid_height > 1:
            self.grid.pop()
        elif height_increment == 1:
            self.grid.append([CellState.EMPTY] * self.grid_width)

        self.grid_width = max(1, self.grid_width + width_increment)
        self.grid_height = max(1, self.grid_height + height_increment)

        self.cell_size = min(
            (self.screen_width - (self.grid_margin_x * 2)) // self.grid_width,
            (self.screen_height - (self.grid_margin_y * 2)) // self.grid_height,
        )

        self.grid_surface = pygame.Surface(
            (self.cell_size * self.grid_width, self.cell_size * self.grid_height)
        )
        self.grid_rect = self.grid_surface.get_rect(
            center=(self.screen_width // 2, self.screen_height // 2)
        )

    def update_cell_state(self, mouse_pos: tuple[int, int], state: CellState):
        mouse_x, mouse_y = mouse_pos

        if not self.grid_rect.collidepoint(mouse_x, mouse_y):
            return

        x = (mouse_x - self.grid_rect.left) // self.cell_size
        y = (mouse_y - self.grid_rect.top) // self.cell_size

        self.grid[y][x] = state

    def draw_grid(self):
        for y, row in enumerate(self.grid):
            for x, cell_state in enumerate(row):
                cell_rect = (
                    self.cell_size * x,
                    self.cell_size * y,
                    self.cell_size,
                    self.cell_size,
                )
                pygame.draw.rect(self.grid_surface, cell_state.value, cell_rect)
                pygame.draw.rect(self.grid_surface, (0, 0, 0), cell_rect, 1)

        self.window.blit(self.grid_surface, self.grid_rect)

    def run(self):
        clock = pygame.time.Clock()

        while True:
            clock.tick(FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN:
                    match event.key:
                        case pygame.K_UP:
                            self.update_grid_dimensions(height_increment=1)
                        case pygame.K_DOWN:
                            self.update_grid_dimensions(height_increment=-1)
                        case pygame.K_LEFT:
                            self.update_grid_dimensions(width_increment=-1)
                        case pygame.K_RIGHT:
                            self.update_grid_dimensions(width_increment=1)
                        case pygame.K_BACKSPACE:
                            self.left_click_draw_mode = (
                                CellState.EMPTY
                                if self.left_click_draw_mode == CellState.WALL
                                else CellState.WALL
                            )
                        case pygame.K_RETURN:
                            grid = [[state for state in row] for row in self.grid]

                            for row in grid:
                                for i, cell_state in enumerate(row):
                                    match cell_state:
                                        case CellState.EMPTY:
                                            row[i] = 0
                                        case CellState.WALL:
                                            row[i] = 1
                                        case CellState.PLAYER:
                                            row[i] = 2
                                        case CellState.ENEMY:
                                            row[i] = 3

                            with open("grid.json", "w") as f:
                                json.dump(grid, f)

            mouse_pressed = pygame.mouse.get_pressed()

            if mouse_pressed[0]:
                self.update_cell_state(
                    pygame.mouse.get_pos(), self.left_click_draw_mode
                )
            elif mouse_pressed[1]:
                self.update_cell_state(pygame.mouse.get_pos(), CellState.PLAYER)
            elif mouse_pressed[2]:
                self.update_cell_state(pygame.mouse.get_pos(), CellState.ENEMY)

            self.window.fill((255, 255, 255))

            self.draw_grid()
            pygame.display.update()
