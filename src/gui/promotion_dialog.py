import pygame
import chess

class PromotionDialog:
    def __init__(self, screen, color):
        self.screen = screen
        self.color = color
        self.font = pygame.font.Font(None, 36)
        self.options = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
        self.images = self._load_promotion_images()
        self.selected = None
        self.done = False

    def _load_promotion_images(self):
        from src.utils.image_loader import load_piece_images
        all_images = load_piece_images()
        color_prefix = 'w' if self.color == chess.WHITE else 'b'
        return {piece: all_images[color_prefix + piece] for piece in ['Q', 'R', 'B', 'N']}

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos
            for i, (piece, img) in enumerate(self.images.items()):
                rect = pygame.Rect(200 + i*80, 300, 70, 70)
                if rect.collidepoint(x, y):
                    self.selected = self.options[i]
                    self.done = True

    def draw(self):
        overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 128))
        self.screen.blit(overlay, (0, 0))

        pygame.draw.rect(self.screen, (255, 255, 255), (150, 250, 500, 150))
        pygame.draw.rect(self.screen, (0, 0, 0), (150, 250, 500, 150), 3)

        text = self.font.render("Выберите фигуру для превращения:", True, (0,0,0))
        self.screen.blit(text, (180, 270))

        for i, (piece, img) in enumerate(self.images.items()):
            scaled = pygame.transform.smoothscale(img, (70, 70))
            self.screen.blit(scaled, (200 + i*80, 300))

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
        return self.selected