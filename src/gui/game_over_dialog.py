import pygame
import textwrap

class GameOverDialog:
    def __init__(self, screen, result_text, reason):
        self.screen = screen
        self.result_text = result_text
        self.reason = reason
        self.font = pygame.font.Font(None, 40)
        self.small_font = pygame.font.Font(None, 30)
        self.done = False
        self.dialog_width = 500
        self.dialog_height = 250
        self.dialog_rect = pygame.Rect(
            (screen.get_width() - self.dialog_width) // 2,
            (screen.get_height() - self.dialog_height) // 2,
            self.dialog_width, self.dialog_height
        )

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN:
            self.done = True

    def draw(self):
        overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 128))
        self.screen.blit(overlay, (0, 0))

        pygame.draw.rect(self.screen, (255, 255, 255), self.dialog_rect)
        pygame.draw.rect(self.screen, (0, 0, 0), self.dialog_rect, 3)

        wrapped_result = textwrap.wrap(self.result_text, width=40)
        y = self.dialog_rect.y + 40
        for line in wrapped_result:
            surf = self.font.render(line, True, (0, 0, 0))
            rect = surf.get_rect(center=(self.dialog_rect.centerx, y))
            self.screen.blit(surf, rect)
            y += 35

        reason_surf = self.small_font.render(f"({self.reason})", True, (100, 100, 100))
        reason_rect = reason_surf.get_rect(center=(self.dialog_rect.centerx, y + 20))
        self.screen.blit(reason_surf, reason_rect)

        instr_surf = self.small_font.render("Нажмите любую клавишу или кликните, чтобы закрыть", True, (0, 0, 0))
        instr_rect = instr_surf.get_rect(center=(self.dialog_rect.centerx, self.dialog_rect.bottom - 30))
        self.screen.blit(instr_surf, instr_rect)

    def run(self):
        clock = pygame.time.Clock()
        while not self.done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                self.handle_event(event)
            self.draw()
            pygame.display.flip()
            clock.tick(30)