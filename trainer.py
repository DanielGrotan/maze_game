import functools
import json
import math
import os
import random
from enum import Enum
from typing import Literal

import neat
import pygame
from neat import ParallelEvaluator

GAMES_PER_GENERATION = 10
MAX_PLAYER_MOVES = 100
PLAYER_STARTING_HEALTH = 100

ENEMY_SHOOT_DAMAGE = 20
ENEMY_HIT_DAMAGE = 50

WALL_CELL_STATE_CODE = 1
PLAYER_CELL_STATE_CODE = 2
ENEMY_CELL_STATE_CODE = 3


class EnemyAction(Enum):
    MOVE_LEFT = 0
    MOVE_RIGHT = 1
    MOVE_UP = 2
    MOVE_DOWN = 3
    STAND_STILL = 4
    SHOOT_PLAYER = 5
    HIT_PLAYER = 6


MovementAmount = Literal[-1] | Literal[0] | Literal[1]


def eval_genomes_display(genomes, config, trainer, player_moves):
    for _, genome in genomes:
        genome.fitness = 0

        trainer.play_game_display(genome, config, player_moves)


def eval_genomes_no_display(genome, config, trainer, player_moves):
    genome.fitness = 0

    trainer.play_game_no_display(genome, config, player_moves)

    return genome.fitness


class Trainer:
    def __init__(
        self, neat_config_path: str, output_directory: str, game_grid_path: str
    ):
        self.neat_config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            neat_config_path,
        )

        self.output_directory = output_directory

        with open(game_grid_path) as f:
            self.game_grid = json.load(f)
            self.game_grid_height = len(self.game_grid)
            self.game_grid_width = len(self.game_grid[0])

            self.find_start_positions()

    def find_start_positions(self):
        self.enemy_start_positions = []

        for y, row in enumerate(self.game_grid):
            for x, cell_state in enumerate(row):
                if cell_state == PLAYER_CELL_STATE_CODE:
                    self.player_start_x, self.player_start_y = x, y
                elif cell_state == ENEMY_CELL_STATE_CODE:
                    self.enemy_start_positions.append((x, y))

    def get_new_position(
        self,
        current_x: int,
        current_y: int,
        *,
        horizontal_movement: MovementAmount = 0,
        vertical_movement: MovementAmount = 0,
    ) -> tuple[int, int]:
        new_x, new_y = current_x, current_y

        # go right
        if horizontal_movement == 1:
            if (
                current_x < self.game_grid_width - 1
                and self.game_grid[current_y][current_x + 1] != WALL_CELL_STATE_CODE
            ):
                new_x += 1
        # go left
        elif horizontal_movement == -1:
            # #print(f"Go left: {current_x = }")
            if (
                current_x > 0
                and self.game_grid[current_y][current_x - 1] != WALL_CELL_STATE_CODE
            ):
                new_x -= 1
        # go down
        elif vertical_movement == 1:
            # going down from the bottom right corner is how the player escapes
            if (
                current_x == self.game_grid_width - 1
                and current_y < self.game_grid_height - 1
            ):
                new_y += 1
            elif (
                current_y < self.game_grid_height - 1
                and self.game_grid[current_y + 1][current_x] != WALL_CELL_STATE_CODE
            ):
                new_y += 1
        # go up
        elif vertical_movement == -1:
            # #print(f"Go up: {current_y = }")
            if (
                current_y > 0
                and self.game_grid[current_y - 1][current_x] != WALL_CELL_STATE_CODE
            ):
                new_y -= 1

        return new_x, new_y

    def get_valid_moves(self, x: int, y: int) -> list[tuple[int, int]]:
        valid_moves = [(x, y)]

        # move left
        # #print(f"Move left: {x = }")
        if x > 0 and self.game_grid[y][x - 1] != WALL_CELL_STATE_CODE:
            valid_moves.append((x - 1, y))

        # move right
        if (
            x < self.game_grid_width - 1
            and self.game_grid[y][x + 1] != WALL_CELL_STATE_CODE
        ):
            valid_moves.append((x + 1, y))

        # move up
        # #print(f"Move up: {y = }")
        if y > 0 and self.game_grid[y - 1][x] != WALL_CELL_STATE_CODE:
            valid_moves.append((x, y - 1))

        # move down from bottom right corner (player wins)
        if x == self.game_grid_width - 1 and y == self.game_grid_height - 1:
            valid_moves.append((x, y + 1))

        # move down
        if (
            y < self.game_grid_height - 1
            and self.game_grid[y + 1][x] != WALL_CELL_STATE_CODE
        ):
            valid_moves.append((x, y + 1))

        return valid_moves

    def generate_player_moves(self) -> list[list[tuple[int, int]]]:
        moves = []

        for _ in range(GAMES_PER_GENERATION):
            current_moves = []

            player_x, player_y = self.player_start_x, self.player_start_y

            for _ in range(MAX_PLAYER_MOVES):
                valid_moves = self.get_valid_moves(player_x, player_y)

                player_x, player_y = random.choice(valid_moves)
                current_moves.append((player_x, player_y))

                if (
                    player_x == self.game_grid_width - 1
                    and player_y == self.game_grid_height
                ):
                    break

            moves.append(current_moves)

        return moves

    def train(self, display_training: bool = False, generations: int = 1000):
        player_moves_path = os.path.join(self.output_directory, "player_moves.json")
        player_moves = None

        if os.path.isfile(player_moves_path):
            with open(player_moves_path) as f:
                player_moves = json.load(f)
        else:
            player_moves = self.generate_player_moves()
            with open(player_moves_path, "w") as f:
                json.dump(player_moves, f)

        checkpoints_path = os.path.join(self.output_directory, "checkpoints")

        latest_checkpoint = -1

        for _, _, filenames in os.walk(checkpoints_path):
            for filename in filenames:
                latest_checkpoint = max(latest_checkpoint, int(filename))

        if latest_checkpoint == -1:
            population = neat.Population(self.neat_config)
        else:
            population = neat.Checkpointer.restore_checkpoint(
                os.path.join(checkpoints_path, str(latest_checkpoint))
            )

        population.add_reporter(neat.StdOutReporter(True))
        population.add_reporter(
            neat.Checkpointer(
                100, None, filename_prefix=os.path.join(checkpoints_path, "")
            )
        )

        if display_training:
            population.run(
                functools.partial(
                    eval_genomes_display, trainer=self, player_moves=player_moves
                ),
                generations,
            )
        else:
            parallel_evaluator = ParallelEvaluator(
                6,
                functools.partial(
                    eval_genomes_no_display, trainer=self, player_moves=player_moves
                ),
            )

            population.run(parallel_evaluator.evaluate, generations)

    def play_game_no_display(
        self, genome, config, player_moves: list[list[tuple[int, int]]]
    ):
        neural_network = neat.nn.FeedForwardNetwork.create(genome, config)

        for moves in player_moves:
            player_health = PLAYER_STARTING_HEALTH
            enemy_positions = [[x, y] for x, y in self.enemy_start_positions]

            moves_made = 0

            for player_x, player_y in moves:
                moves_made += 1

                if (
                    player_x == self.game_grid_width - 1
                    and player_y == self.game_grid_height
                ):
                    break

                for enemy_index, (enemy_x, enemy_y) in enumerate(enemy_positions):
                    network_inputs = [
                        (enemy_index + 1) / len(enemy_positions),
                        (player_x + 1) / self.game_grid_width,
                        (player_y + 1) / self.game_grid_height,
                        player_health / PLAYER_STARTING_HEALTH,
                    ]

                    for (
                        x,
                        y,
                    ) in enemy_positions:
                        network_inputs.append((x + 1) / self.game_grid_width)
                        network_inputs.append((y + 1) / self.game_grid_height)

                    output = neural_network.activate(network_inputs)

                    largest_value = -1
                    decision = -1

                    for i, value in enumerate(output):
                        # #print(f"Decision: {value = }")
                        if value > largest_value:
                            largest_value = value
                            decision = i

                    decision = EnemyAction(decision)

                    match decision:
                        case EnemyAction.MOVE_LEFT:
                            enemy_x, enemy_y = self.get_new_position(
                                enemy_x, enemy_y, horizontal_movement=-1
                            )
                        case EnemyAction.MOVE_RIGHT:
                            enemy_x, enemy_y = self.get_new_position(
                                enemy_x, enemy_y, horizontal_movement=1
                            )
                        case EnemyAction.MOVE_UP:
                            enemy_x, enemy_y = self.get_new_position(
                                enemy_x, enemy_y, vertical_movement=-1
                            )
                        case EnemyAction.MOVE_DOWN:
                            enemy_x, enemy_y = self.get_new_position(
                                enemy_x, enemy_y, vertical_movement=1
                            )
                        case EnemyAction.STAND_STILL:
                            pass
                        case EnemyAction.SHOOT_PLAYER:
                            distance_to_player = math.sqrt(
                                (player_x - enemy_x) ** 2 + (player_y - enemy_y) ** 2
                            )
                            if distance_to_player <= 3:
                                # player horizontally aligned
                                if enemy_y == player_y:
                                    # #print(f"Enemy to the right: {enemy_x = }")
                                    # player to the right of enemy
                                    if enemy_x <= player_x and all(
                                        self.game_grid[enemy_y][x]
                                        != WALL_CELL_STATE_CODE
                                        for x in range(enemy_x, player_x)
                                    ):
                                        player_health -= ENEMY_SHOOT_DAMAGE
                                    # enemy to the right of player
                                    elif enemy_x > player_x and all(
                                        self.game_grid[enemy_y][x]
                                        != WALL_CELL_STATE_CODE
                                        for x in range(player_x, enemy_x)
                                    ):
                                        player_health -= ENEMY_SHOOT_DAMAGE
                                # vertically aligned
                                if enemy_x == player_x:
                                    # #print(f"Enemy below: {enemy_y = }")
                                    # player below enemy
                                    if enemy_y <= player_y and all(
                                        self.game_grid[y][enemy_x]
                                        != WALL_CELL_STATE_CODE
                                        for y in range(enemy_y, player_y)
                                    ):
                                        player_health -= ENEMY_SHOOT_DAMAGE
                                    # enemy below player
                                    elif enemy_y > player_y and all(
                                        self.game_grid[y][enemy_x]
                                        != WALL_CELL_STATE_CODE
                                        for y in range(player_y, enemy_y)
                                    ):
                                        player_health -= ENEMY_SHOOT_DAMAGE
                        case EnemyAction.HIT_PLAYER:
                            distance_to_player = math.sqrt(
                                (player_x - enemy_x) ** 2 + (player_y - enemy_y) ** 2
                            )
                            if distance_to_player <= 1:
                                player_health -= ENEMY_HIT_DAMAGE

                    if enemy_positions[enemy_index] != [enemy_x, enemy_y]:
                        genome.fitness += 0.1

                    enemy_positions[enemy_index] = [enemy_x, enemy_y]  # type: ignore

                    genome.fitness -= (
                        math.sqrt((player_x - enemy_x) ** 2 + (player_y - enemy_y) ** 2)
                        / MAX_PLAYER_MOVES**2
                    )

                    if player_health <= 0:
                        break

                if player_health <= 0:
                    break

            # calculate fitness
            # #print(f"Fitness calculation: {player_health = }")
            if player_health > 0:
                genome.fitness -= (MAX_PLAYER_MOVES - moves_made) ** 2 + (
                    PLAYER_STARTING_HEALTH - player_health
                )
            else:
                genome.fitness += (MAX_PLAYER_MOVES - moves_made) ** 2 + (
                    PLAYER_STARTING_HEALTH - player_health
                )

    def play_game_display(
        self, genome, config, player_moves: list[list[tuple[int, int]]]
    ):
        neural_network = neat.nn.FeedForwardNetwork.create(genome, config)

        window = pygame.display.set_mode((800, 600))

        cell_size = min(
            (800 - 50 * 2) // self.game_grid_width,
            (600 - 50 * 2) // self.game_grid_height,
        )

        grid_surface = pygame.Surface(
            (cell_size * self.game_grid_width, cell_size * self.game_grid_height)
        )
        grid_rect = grid_surface.get_rect(center=(800 // 2, 600 // 2))

        pygame.font.init()
        font = pygame.font.SysFont("Helvetica", 24)

        clock = pygame.time.Clock()

        for moves in player_moves:
            player_health = PLAYER_STARTING_HEALTH
            enemy_positions = [[x, y] for x, y in self.enemy_start_positions]

            moves_made = 0

            for player_x, player_y in moves:
                clock.tick(20)
                moves_made += 1

                window.fill((255, 255, 255))

                for y, row in enumerate(self.game_grid):
                    for x, cell_state in enumerate(row):
                        if cell_state == WALL_CELL_STATE_CODE:
                            pygame.draw.rect(
                                grid_surface,
                                (0, 0, 0),
                                (cell_size * x, cell_size * y, cell_size, cell_size),
                            )
                        else:
                            pygame.draw.rect(
                                grid_surface,
                                (70, 70, 70),
                                (cell_size * x, cell_size * y, cell_size, cell_size),
                            )

                pygame.draw.rect(
                    grid_surface,
                    (0, 255, 0),
                    (cell_size * player_x, cell_size * player_y, cell_size, cell_size),
                )

                text = font.render(str(player_health), True, (0, 0, 0))
                text_rect = text.get_rect(
                    center=(
                        cell_size * player_x + cell_size // 2,
                        cell_size * player_y + cell_size // 2,
                    )
                )
                grid_surface.blit(text, text_rect)

                if (
                    player_x == self.game_grid_width - 1
                    and player_y == self.game_grid_height
                ):
                    break

                for enemy_index, (enemy_x, enemy_y) in enumerate(enemy_positions):
                    network_inputs = [
                        (enemy_index + 1) / len(enemy_positions),
                        (player_x + 1) / self.game_grid_width,
                        (player_y + 1) / self.game_grid_height,
                        player_health / PLAYER_STARTING_HEALTH,
                    ]

                    for (
                        x,
                        y,
                    ) in enemy_positions:
                        network_inputs.append((x + 1) / self.game_grid_width)
                        network_inputs.append((y + 1) / self.game_grid_height)

                    output = neural_network.activate(network_inputs)

                    largest_value = -1
                    decision = -1

                    for i, value in enumerate(output):
                        # #print(f"Decision: {value = }")
                        if value > largest_value:
                            largest_value = value
                            decision = i

                    decision = EnemyAction(decision)

                    match decision:
                        case EnemyAction.MOVE_LEFT:
                            enemy_x, enemy_y = self.get_new_position(
                                enemy_x, enemy_y, horizontal_movement=-1
                            )
                        case EnemyAction.MOVE_RIGHT:
                            enemy_x, enemy_y = self.get_new_position(
                                enemy_x, enemy_y, horizontal_movement=1
                            )
                        case EnemyAction.MOVE_UP:
                            enemy_x, enemy_y = self.get_new_position(
                                enemy_x, enemy_y, vertical_movement=-1
                            )
                        case EnemyAction.MOVE_DOWN:
                            enemy_x, enemy_y = self.get_new_position(
                                enemy_x, enemy_y, vertical_movement=1
                            )
                        case EnemyAction.STAND_STILL:
                            pass
                        case EnemyAction.SHOOT_PLAYER:
                            distance_to_player = math.sqrt(
                                (player_x - enemy_x) ** 2 + (player_y - enemy_y) ** 2
                            )
                            if distance_to_player <= 3:
                                # player horizontally aligned
                                if enemy_y == player_y:
                                    # #print(f"Enemy to the right: {enemy_x = }")
                                    # player to the right of enemy
                                    if enemy_x <= player_x and all(
                                        self.game_grid[enemy_y][x]
                                        != WALL_CELL_STATE_CODE
                                        for x in range(enemy_x, player_x)
                                    ):
                                        player_health -= ENEMY_SHOOT_DAMAGE
                                    # enemy to the right of player
                                    elif enemy_x > player_x and all(
                                        self.game_grid[enemy_y][x]
                                        != WALL_CELL_STATE_CODE
                                        for x in range(player_x, enemy_x)
                                    ):
                                        player_health -= ENEMY_SHOOT_DAMAGE
                                # vertically aligned
                                if enemy_x == player_x:
                                    # #print(f"Enemy below: {enemy_y = }")
                                    # player below enemy
                                    if enemy_y <= player_y and all(
                                        self.game_grid[y][enemy_x]
                                        != WALL_CELL_STATE_CODE
                                        for y in range(enemy_y, player_y)
                                    ):
                                        player_health -= ENEMY_SHOOT_DAMAGE
                                    # enemy below player
                                    elif enemy_y > player_y and all(
                                        self.game_grid[y][enemy_x]
                                        != WALL_CELL_STATE_CODE
                                        for y in range(player_y, enemy_y)
                                    ):
                                        player_health -= ENEMY_SHOOT_DAMAGE
                        case EnemyAction.HIT_PLAYER:
                            distance_to_player = math.sqrt(
                                (player_x - enemy_x) ** 2 + (player_y - enemy_y) ** 2
                            )
                            if distance_to_player <= 1:
                                player_health -= ENEMY_HIT_DAMAGE

                    enemy_positions[enemy_index] = [enemy_x, enemy_y]  # type: ignore

                    if player_health <= 0:
                        break

                for enemy_x, enemy_y in enemy_positions:
                    pygame.draw.rect(
                        grid_surface,
                        (255, 0, 0),
                        (
                            cell_size * enemy_x,
                            cell_size * enemy_y,
                            cell_size,
                            cell_size,
                        ),
                    )

                window.blit(grid_surface, grid_rect)

                pygame.display.update()

                if player_health <= 0:
                    break

            # calculate fitness
            # #print(f"Fitness calculation: {player_health = }")
            if player_health > 0:
                genome.fitness -= (MAX_PLAYER_MOVES - moves_made) ** 2 + (
                    PLAYER_STARTING_HEALTH - player_health
                )
            else:
                genome.fitness += (MAX_PLAYER_MOVES - moves_made) ** 2 + (
                    PLAYER_STARTING_HEALTH - player_health
                )


def main():
    trainer = Trainer("neat-config", "training_output", "grid.json")
    trainer.train()


if __name__ == "__main__":
    main()
