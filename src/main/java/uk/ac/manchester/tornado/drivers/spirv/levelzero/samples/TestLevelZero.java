package uk.ac.manchester.tornado.drivers.spirv.levelzero.samples;

import uk.ac.manchester.tornado.drivers.spirv.levelzero.*;

import java.io.IOException;

public class TestLevelZero {

    // Test Program
    public static void main(String[] args) throws IOException {
        System.out.println("Level-ZERO JNI Library - Test");

        // Create the Level Zero Driver
        LevelZeroDriver driver = new LevelZeroDriver();
        int result = driver.zeInit(ZeInitFlag.ZE_INIT_FLAG_GPU_ONLY);
        LevelZeroUtils.errorLog("zeInit", result);

        int[] numDrivers = new int[1];
        result = driver.zeDriverGet(numDrivers, null);
        LevelZeroUtils.errorLog("zeDriverGet", result);

        ZeDriverHandle driverHandler = new ZeDriverHandle(numDrivers[0]);

        result = driver.zeDriverGet(numDrivers, driverHandler);
        LevelZeroUtils.errorLog("zeDriverGet", result);

        ZeDeviceType type = ZeDeviceType.ZE_DEVICE_TYPE_GPU;

        // ============================================
        // Create the Context
        // ============================================
        // Create context Description
        ZeContextDesc contextDescription = new ZeContextDesc();
        // Create context object
        LevelZeroContext context = new LevelZeroContext(driverHandler, contextDescription);
        // Call native method for creating the context
        result = context.zeContextCreate(driverHandler.getZe_driver_handle_t_ptr()[0], 0);
        LevelZeroUtils.errorLog("zeContextCreate", result);

        // Get number of devices in a driver
        int[] deviceCount = new int[1];
        result = driver.zeDeviceGet(driverHandler, 0, deviceCount, null);
        LevelZeroUtils.errorLog("zeDeviceGet", result);

        // Instantiate a device Handler
        ZeDevicesHandle deviceHandler = new ZeDevicesHandle(deviceCount[0]);
        result = driver.zeDeviceGet(driverHandler, 0, deviceCount, deviceHandler);
        LevelZeroUtils.errorLog("zeDeviceGet", result);

        // ============================================
        // Query driver properties
        // ============================================
        ZeDriverProperties driverProperties = new ZeDriverProperties(Ze_Structure_Type.ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES);
        result = driver.zeDriverGetProperties(driverHandler, 0, driverProperties);
        LevelZeroUtils.errorLog("zeDriverGetProperties", result);

        System.out.println("Driver Version: " + driverProperties.getDriverVersion());

        ZeAPIVersion apiVersion = new ZeAPIVersion();
        result = driver.zeDriverGetApiVersion(driverHandler, 0, apiVersion);
        LevelZeroUtils.errorLog("zeDriverGetApiVersion", result);

        System.out.println("Level Zero API Version: " + apiVersion);

        // ============================================
        // Query device properties
        // ============================================
        LevelZeroDevice device = driver.getDevice(driverHandler, 0);
        ZeDeviceProperties deviceProperties = new ZeDeviceProperties();
        result = device.zeDeviceGetProperties(device.getDeviceHandlerPtr(), deviceProperties);
        LevelZeroUtils.errorLog("zeDeviceGetProperties", result);
        System.out.println(deviceProperties);

        // ============================================
        // Query device compute-properties
        // ============================================
        ZeComputeProperties computeProperties = new ZeComputeProperties();
        result = device.zeDeviceGetComputeProperties(device.getDeviceHandlerPtr(), computeProperties);
        LevelZeroUtils.errorLog("zeDeviceGetComputeProperties", result);
        System.out.println(computeProperties);

        // ============================================
        // Query device memory
        // ============================================
        // A) Count memories
        int[] memoryCount = new int[1];
        result = device.zeDeviceGetMemoryProperties(device.getDeviceHandlerPtr(), memoryCount, null);
        LevelZeroUtils.errorLog("zeDeviceGetMemoryProperties", result);

        // B) Access the properties of each of the memories
        ZeMemoryProperties[] memoryProperties = new ZeMemoryProperties[memoryCount[0]];
        result = device.zeDeviceGetMemoryProperties(device.getDeviceHandlerPtr(), memoryCount, memoryProperties);
        LevelZeroUtils.errorLog("zeDeviceGetMemoryProperties", result);
        for (ZeMemoryProperties m : memoryProperties) {
            System.out.println(m);
        }

        // ============================================
        // Query device memory access properties
        // ============================================
        ZeMemoryAccessProperties memoryAccessProperties = new ZeMemoryAccessProperties();
        result = device.zeDeviceGetMemoryAccessProperties(device.getDeviceHandlerPtr(), memoryAccessProperties);
        LevelZeroUtils.errorLog("zeDeviceGetMemoryAccessProperties", result);
        System.out.println(memoryAccessProperties);

        // ============================================
        // Query device cache properties
        // ============================================
        int[] cacheCount = new int[1];
        result = device.zeDeviceGetCacheProperties(device.getDeviceHandlerPtr(), cacheCount, null);
        LevelZeroUtils.errorLog("zeDeviceGetCacheProperties", result);

        ZeDeviceCacheProperties[] cacheProperties = new ZeDeviceCacheProperties[cacheCount[0]];
        result = device.zeDeviceGetCacheProperties(device.getDeviceHandlerPtr(), cacheCount, cacheProperties);
        LevelZeroUtils.errorLog("zeDeviceGetCacheProperties", result);

        for (ZeDeviceCacheProperties c : cacheProperties) {
            System.out.println(c);
        }

        // ============================================
        // Query device image properties
        // ============================================
        ZeDeviceImageProperties imageProperties = new ZeDeviceImageProperties();
        device.zeDeviceGetImageProperties(device.getDeviceHandlerPtr(), imageProperties);
        LevelZeroUtils.errorLog("zeDeviceGetImageProperties", result);
        System.out.println(imageProperties);

        // ============================================
        // Create a command queue
        // ============================================
        // A) Get the number of command queue groups
        int[] numQueueGroups = new int[1];
        result = device.zeDeviceGetCommandQueueGroupProperties(device.getDeviceHandlerPtr(), numQueueGroups, null);
        LevelZeroUtils.errorLog("zeDeviceGetCommandQueueGroupProperties", result);

        if (numQueueGroups[0] == 0) {
            throw new RuntimeException("Number of Queue Groups is 0 for device: " + device.getDeviceProperties().getName());
        }

        ZeCommandQueueGroupProperties[] commandQueueGroupProperties = new ZeCommandQueueGroupProperties[numQueueGroups[0]];
        result = device.zeDeviceGetCommandQueueGroupProperties(device.getDeviceHandlerPtr(), numQueueGroups, commandQueueGroupProperties);
        LevelZeroUtils.errorLog("zeDeviceGetCommandQueueGroupProperties", result);
        for (ZeCommandQueueGroupProperties p : commandQueueGroupProperties) {
            System.out.println(p);
        }

        ZeCommandQueueHandle commandQueueHandle = new ZeCommandQueueHandle();
        LevelZeroCommandQueue commandQueue = new LevelZeroCommandQueue(context, commandQueueHandle);
        ZeCommandQueueDescription commandQueueDescription = new ZeCommandQueueDescription();

        for (int i = 0; i < numQueueGroups[0]; i++) {
            if ((commandQueueGroupProperties[i].getFlags()
                    & ZeCommandQueueGroupPropertyFlags.ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) == ZeCommandQueueGroupPropertyFlags.ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) {
                commandQueueDescription.setOrdinal(i);
            }
        }

        // B) Create the command queue via the context
        commandQueueDescription.setIndex(0);
        commandQueueDescription.setMode(ZeCommandQueueMode.ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS);
        // zeCommandQueueCreate(context, device, &cmdQueueDesc, &cmdQueue);
        result = context.zeCommandQueueCreate(context.getContextHandle().getContextPtr()[0], device.getDeviceHandlerPtr(), commandQueueDescription, commandQueueHandle);
        LevelZeroUtils.errorLog("zeCommandQueueCreate", result);

        // ============================================
        // Create a command list
        // ============================================
        ZeCommandQueueListHandle zeCommandListHandler = new ZeCommandQueueListHandle();
        LevelZeroCommandList commandList = new LevelZeroCommandList(context, zeCommandListHandler);
        ZeCommandListDescription commandListDescription = new ZeCommandListDescription();
        commandListDescription.setCommandQueueGroupOrdinal(commandQueueDescription.getOrdinal());
        result = context.zeCommandListCreate(context.getContextHandle().getContextPtr()[0], device.getDeviceHandlerPtr(), commandListDescription, zeCommandListHandler);
        LevelZeroUtils.errorLog("zeCommandListCreate", result);

        final int elements = 8192;
        final int bufferSize = elements * 4;
        ZeDeviceMemAllocDesc deviceMemAllocDesc = new ZeDeviceMemAllocDesc();
        deviceMemAllocDesc.setFlags(ZeDeviceMemAllocFlags.ZE_DEVICE_MEM_ALLOC_FLAG_BIAS_UNCACHED);
        deviceMemAllocDesc.setOrdinal(0);

        ZeHostMemAllocDesc hostMemAllocDesc = new ZeHostMemAllocDesc();
        hostMemAllocDesc.setFlags(ZeHostMemAllocFlags.ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED);

        LevelZeroBufferInteger bufferA = new LevelZeroBufferInteger();
        result = context.zeMemAllocShared(context.getContextHandle().getContextPtr()[0], deviceMemAllocDesc, hostMemAllocDesc, bufferSize, 1, device.getDeviceHandlerPtr(), bufferA);
        LevelZeroUtils.errorLog("zeMemAllocShared", result);

        LevelZeroBufferInteger bufferB = new LevelZeroBufferInteger();
        result = context.zeMemAllocShared(context.getDefaultContextPtr(), deviceMemAllocDesc, hostMemAllocDesc, bufferSize, 1, device.getDeviceHandlerPtr(), bufferB);
        LevelZeroUtils.errorLog("zeMemAllocShared", result);

        bufferA.memset(100, elements);
        bufferB.memset(0, elements);

        ZeModuleHandle module = new ZeModuleHandle();
        ZeModuleDesc moduleDesc = new ZeModuleDesc();
        ZeBuildLogHandle buildLog = new ZeBuildLogHandle();
        moduleDesc.setFormat(ZeModuleFormat.ZE_MODULE_FORMAT_IL_SPIRV);
        moduleDesc.setBuildFlags("");

        LevelZeroBinaryModule binaryModule = new LevelZeroBinaryModule("/tmp/example.spv");
        result = binaryModule.readBinary();
        LevelZeroUtils.errorLog("readBinary", result);

        result = context.zeModuleCreate(context.getDefaultContextPtr(), device.getDeviceHandlerPtr(), binaryModule, moduleDesc, module, buildLog);
        LevelZeroUtils.errorLog("zeModuleCreate", result);

        if (result != ZeResult.ZE_RESULT_SUCCESS) {
            // Print Logs
            int[] sizeLog = new int[1];
            String errorMessage = new String();
            result = context.zeModuleBuildLogGetString(buildLog, sizeLog, errorMessage);
            System.out.println("LOGS::: " + sizeLog[0] + "  -- " + errorMessage);
            LevelZeroUtils.errorLog("zeModuleBuildLogGetString", result);
        }

        // Create Module Object
        LevelZeroModule levelZeroModule = new LevelZeroModule(module, moduleDesc, buildLog);

        // Destroy Log
        result = levelZeroModule.zeModuleBuildLogDestroy(buildLog);
        LevelZeroUtils.errorLog("zeModuleBuildLogDestroy", result);

        ZeKernelDesc kernelDesc = new ZeKernelDesc();
        ZeKernelHandle kernel = new ZeKernelHandle();
        kernelDesc.setKernelName("copydata");
        result = levelZeroModule.zeKernelCreate(module.getPtrZeModuleHandle(), kernelDesc, kernel);
        LevelZeroUtils.errorLog("zeKernelCreate", result);

        // We create a kernel Object
        LevelZeroKernel levelZeroKernel = new LevelZeroKernel(kernelDesc, kernel);

        // Prepare kernel for launch
        // A) Suggest scheduling parameters to level-zero
        int[] groupSizeX = new int[] { 32 };
        int[] groupSizeY = new int[] { 1 };
        int[] groupSizeZ = new int[] { 1 };
        result = levelZeroKernel.zeKernelSuggestGroupSize(kernel.getPtrZeKernelHandle(), elements, 1, 1, groupSizeX, groupSizeY, groupSizeZ);
        LevelZeroUtils.errorLog("zeKernelSuggestGroupSize", result);

        result = levelZeroKernel.zeKernelSetGroupSize(kernel.getPtrZeKernelHandle(), groupSizeX, groupSizeY, groupSizeZ);
        LevelZeroUtils.errorLog("zeKernelSetGroupSize", result);

        result = levelZeroKernel.zeKernelSetArgumentValue(kernel.getPtrZeKernelHandle(), 0, Sizeof.POINTER.getNumBytes(), bufferA);
        result |= levelZeroKernel.zeKernelSetArgumentValue(kernel.getPtrZeKernelHandle(), 1, Sizeof.POINTER.getNumBytes(), bufferB);
        LevelZeroUtils.errorLog("zeKernelSetArgumentValue", result);

        // Dispatch SPIR-V Kernel
        ZeGroupDispatch dispatch = new ZeGroupDispatch();
        dispatch.setGroupCountX(elements / groupSizeX[0]);
        dispatch.setGroupCountY(1);
        dispatch.setGroupCountZ(1);

        // Launch the kernel on the Intel Integrated GPU
        result = commandList.zeCommandListAppendLaunchKernel(zeCommandListHandler.getPtrZeCommandListHandle(), kernel.getPtrZeKernelHandle(), dispatch, null, 0, null);
        LevelZeroUtils.errorLog("zeCommandListAppendLaunchKernel", result);

        result = commandList.zeCommandListClose(zeCommandListHandler.getPtrZeCommandListHandle());
        LevelZeroUtils.errorLog("zeCommandListClose", result);

        result = commandQueue.zeCommandQueueExecuteCommandLists(commandQueueHandle.getCommandQueueHandlerPointer(), 1, zeCommandListHandler.getPtrZeCommandListHandle(), null);
        LevelZeroUtils.errorLog("zeCommandQueueExecuteCommandLists", result);

        result = commandQueue.zeCommandQueueSynchronize(commandQueueHandle.getCommandQueueHandlerPointer(), Long.MAX_VALUE);
        LevelZeroUtils.errorLog("zeCommandQueueSynchronize", result);

        boolean isEqual = bufferA.isEqual(bufferB, elements);
        if (isEqual) {
            System.out.println("Result is correct");
        } else {
            System.out.println("Result is wrong");
        }

        result = context.zeMemFree(context.getDefaultContextPtr(), bufferA);
        result |= context.zeMemFree(context.getDefaultContextPtr(), bufferB);
        LevelZeroUtils.errorLog("zeMemFree", result);

        result = context.zeCommandListDestroy(zeCommandListHandler);
        LevelZeroUtils.errorLog("zeCommandListDestroy", result);

        result = context.zeCommandQueueDestroy(commandQueueHandle);
        LevelZeroUtils.errorLog("zeCommandQueueDestroy", result);

        result = driver.zeContextDestroy(context);
        LevelZeroUtils.errorLog("zeContextDestroy", result);

    }
}
