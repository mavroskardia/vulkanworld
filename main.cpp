#include <iostream>
#include <cstdlib>
#include "VulkanRenderer.h"
#include "main.h"

const uint32_t WIDTH = 1024;
const uint32_t HEIGHT = 768;

int main(int argc, char** argv) {

	if (SDL_Init(SDL_INIT_EVERYTHING) != 0) {
		std::cerr << SDL_GetError() << std::endl;
		return EXIT_FAILURE;
	}

	atexit(SDL_Quit);

	SDL_Window* window;
	SDL_Renderer* renderer;
	SDL_Event evt;

	if (SDL_CreateWindowAndRenderer(WIDTH, HEIGHT, SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE, &window, &renderer) != 0) {
		std::cerr << SDL_GetError() << std::endl;
		return EXIT_FAILURE;
	}

	VulkanRenderer vkrenderer;
	vkrenderer.initVulkan(window);
		
	bool running = true;

	while (running) {

		while (SDL_PollEvent(&evt)) {
			switch (evt.type) {
				case SDL_QUIT:
					running = false;
					break;
				case SDL_KEYDOWN:
					if (evt.key.keysym.sym == SDLK_ESCAPE) {
						running = false;
					}
					break;
				case SDL_MOUSEWHEEL:					
					vkrenderer.deltaZoom(evt.wheel.y);
					break;
			}
		}

		vkrenderer.drawFrame(window);

	}
	
	vkrenderer.cleanup();
	SDL_DestroyRenderer(renderer);
	SDL_DestroyWindow(window);

	return EXIT_SUCCESS;
}