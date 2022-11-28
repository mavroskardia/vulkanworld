#pragma once

#ifndef __VULKANRENDERER_H_
#define __VULKANRENDERER_H_

#include <fstream>
#include <optional>
#include <array>
#include <chrono>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtx/hash.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <vulkan/vulkan.h>
#include <SDL.h>
#include <SDL_vulkan.h>

const int MAX_FRAMES_IN_FLIGHT = 2;

const std::string MODEL_PATH = "models/viking_room.obj";
const std::string TEXTURE_PATH = "textures/viking_room.png";


struct UniformBufferObject {
	alignas(16) glm::mat4 model;
	alignas(16) glm::mat4 view;
	alignas(16) glm::mat4 proj;
};

struct Vertex {

	glm::vec3 pos;
	glm::vec3 normal;
	glm::vec3 color;
	glm::vec2 texCoord;

	bool operator==(const Vertex& other) const {
		return pos == other.pos && color == other.color && texCoord == other.texCoord;
	}

	static VkVertexInputBindingDescription getBindingDescription() {
		
		VkVertexInputBindingDescription bindingDescription {
			.binding = 0,
			.stride = sizeof(Vertex),
			.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
		};

		return bindingDescription;
	}

	static std::array<VkVertexInputAttributeDescription, 4> getAttributeDescriptions() {
		
		std::array<VkVertexInputAttributeDescription, 4> attributeDescriptions{};

		attributeDescriptions[0] = {
			.location = 0,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(Vertex, pos),
		};

		attributeDescriptions[1] = {
			.location = 1,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(Vertex, color),
		};

		attributeDescriptions[2] = {
			.location = 2,
			.binding = 0,
			.format = VK_FORMAT_R32G32_SFLOAT,
			.offset = offsetof(Vertex, texCoord),
		};

		attributeDescriptions[3] = {
			.location = 3,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(Vertex, normal),
		};

		return attributeDescriptions;
	}

};

namespace std {
	template<> struct hash<Vertex> {
		size_t operator()(Vertex const& vertex) const {
			return ((hash<glm::vec3>()(vertex.pos) ^
				(hash<glm::vec3>()(vertex.normal) << 1) >> 1) ^
				(hash<glm::vec3>()(vertex.color) << 1)) >> 1 ^
				(hash<glm::vec2>()(vertex.texCoord) << 1);
		}
	};
}

struct QueueFamilyIndices {
	std::optional<uint32_t> graphicsFamily;
	std::optional<uint32_t> presentFamily;

	bool isComplete() {
		return graphicsFamily.has_value() && presentFamily.has_value();
	}
};

struct SwapChainSupportDetails {
	VkSurfaceCapabilitiesKHR capabilities;
	std::vector<VkSurfaceFormatKHR> formats;
	std::vector<VkPresentModeKHR> presentModes;
};

class VulkanRenderer {
public:
	void deltaZoom(int32_t delta);
	void initVulkan(SDL_Window* window);
	void setupDebugMessenger(void);
	void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo);
	VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger);
	void createInstance(SDL_Window* window);
	void createLogicalDevice(void);
	
	void createSwapChain(SDL_Window* window);
	void recreateSwapChain(SDL_Window* window);
	void cleanupSwapChain(void);
	
	void cleanup(void);
	void pickPhysicalDevice(void);
	void createImageViews(void);
	void createDescriptorSetLayout(void);
	void createGraphicsPipeline(void);
	void createRenderPass(void);
	void createFramebuffers(void);
	void createUniformBuffers(void);
	void createDescriptorPool(void);
	void createDescriptorSets(void);
	void createTextureImage(void);
	void createTextureImageView(void);
	void createTextureSampler(void);
	void createIndexBuffer(void);
	void createVertexBuffer(void);
	void createCommandPool(void);
	void createCommandBuffers(void);
	void createDepthResources(void);
	void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex);	

	void updateUniformBuffer(uint32_t currentImage);

	void createSyncObjects(void);

	void drawFrame(SDL_Window* window);

	VkShaderModule createShaderModule(const std::vector<char>& code);
	
	QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device, VkSurfaceKHR surface);
	bool isDeviceSuitable(VkPhysicalDevice);
	bool checkDeviceExtensionsSupport(VkPhysicalDevice device);
	bool checkValidationLayerSupport();
	SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device);
	VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);
	VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes);
	VkExtent2D chooseSwapExtent(SDL_Window* window, const VkSurfaceCapabilitiesKHR& capabilities);

	void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);
	uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
	void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory);
	
	void createImage(uint32_t width, uint32_t height, uint32_t mipLevels, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory);
	VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags, uint32_t mipLevels);
	void generateMipmaps(VkImage image, VkFormat imageFormat, int32_t textWidth, int32_t texHeight, uint32_t mipLevels);

	VkCommandBuffer beginSingleTimeCommands(void);
	void endSingleTimeCommands(VkCommandBuffer commandBuffer);
	void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t mipLevels);
	void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);
	VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features);
	VkFormat findDepthFormat(void);
	bool hasStencilComponent(VkFormat format);
	
	void loadModel(void);

private:

	float zoom = 0.0f;

	VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
	VkInstance instance;
	VkSurfaceKHR surface;
	VkDevice device;

	VkDebugUtilsMessengerEXT debugMessenger;

	VkQueue graphicsQueue;
	VkQueue presentQueue;
	VkRenderPass renderPass;
	VkPipelineLayout pipelineLayout;
	VkPipeline graphicsPipeline;

	uint32_t currentFrame = 0;
	VkCommandPool commandPool;
	std::vector<VkCommandBuffer> commandBuffers;
	
	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;

	VkBuffer vertexBuffer;
	VkDeviceMemory vertexBufferMemory;
	VkBuffer indexBuffer;
	VkDeviceMemory indexBufferMemory;

	VkDescriptorSetLayout descriptorSetLayout;
	VkDescriptorPool descriptorPool;
	std::vector<VkDescriptorSet> descriptorSets;

	std::vector<VkBuffer> uniformBuffers;
	std::vector<VkDeviceMemory> uniformBuffersMemory;
	std::vector<void*> uniformBuffersMapped;

	std::vector<VkSemaphore> imageAvailableSemaphores;
	std::vector<VkSemaphore> renderFinishedSemaphores;
	std::vector<VkFence> inFlightFences;

	const VkClearValue clearColor = { { { 0.0f, 0.0f, 0.0f, 1.0f } } };

	std::vector<VkFramebuffer> swapChainFramebuffers;	
	VkSwapchainKHR swapChain;
	std::vector<VkImage> swapChainImages;
	VkFormat swapChainImageFormat;
	VkExtent2D swapChainExtent;
	std::vector<VkImageView> swapChainImageViews;

	uint32_t mipLevels;
	VkImage textureImage;
	VkDeviceMemory textureImageMemory;
	VkImageView textureImageView;
	VkSampler textureSampler;

	VkImage depthImage;
	VkDeviceMemory depthImageMemory;
	VkImageView depthImageView;


	const std::vector<const char*> deviceExtensions = {
		VK_KHR_SWAPCHAIN_EXTENSION_NAME
	};

	const std::vector<const char*> validationLayers = {
		"VK_LAYER_KHRONOS_validation"
	};

#ifdef NDEBUG
	const bool enableValidationLayers = false;
#else
	const bool enableValidationLayers = true;
#endif

};

static std::vector<char> readFile(const std::string& filename) {
	std::ifstream file(filename, std::ios::ate | std::ios::binary);

	if (!file.is_open()) {
		throw std::runtime_error("failed to open file!");
	}

	size_t fileSize = (size_t) file.tellg();
	std::vector<char> buffer(fileSize);

	file.seekg(0);
	file.read(buffer.data(), fileSize);

	file.close();

	return buffer;
}

#endif