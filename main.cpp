#include <iostream>
#include <cstdlib>
#include <vulkan/vulkan.hpp>
#include <SDL.h>

#include "VulkanRenderer.h"


int main(int argc, char** argv) {

	if (SDL_Init(SDL_INIT_EVERYTHING) != 0) {
		std::cerr << SDL_GetError() << std::endl;
		return EXIT_FAILURE;
	}

	atexit(SDL_Quit);

	SDL_Window* window;
	SDL_Renderer* renderer;
	SDL_Event evt;

	if (SDL_CreateWindowAndRenderer(1024, 768, SDL_WINDOW_VULKAN, &window, &renderer) != 0) {
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
			}
		}

		SDL_SetRenderDrawColor(renderer, 100, 100, 0, 255);
		SDL_RenderClear(renderer);

		// drawing operations here

		SDL_RenderPresent(renderer);

		SDL_Delay(17); // 17 == ~60fps (1000/17 ~= 60)
	}
	
	SDL_DestroyRenderer(renderer);
	SDL_DestroyWindow(window);
	vkrenderer.cleanup();

	return EXIT_SUCCESS;
}