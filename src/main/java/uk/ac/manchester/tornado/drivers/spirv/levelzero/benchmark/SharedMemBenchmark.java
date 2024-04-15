package uk.ac.manchester.tornado.drivers.spirv.levelzero.benchmark;

import uk.ac.manchester.tornado.drivers.spirv.levelzero.LevelZeroBufferInteger;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.LevelZeroByteBuffer;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.LevelZeroCommandList;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.LevelZeroCommandQueue;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.LevelZeroContext;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.LevelZeroDevice;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.LevelZeroDriver;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.LevelZeroKernel;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.LevelZeroModule;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.Sizeof;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.ZeAPIVersion;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.ZeBuildLogHandle;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.ZeCommandListDescriptor;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.ZeCommandListHandle;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.ZeCommandQueueDescriptor;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.ZeCommandQueueGroupProperties;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.ZeCommandQueueGroupPropertyFlags;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.ZeCommandQueueHandle;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.ZeCommandQueueMode;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.ZeContextDescriptor;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.ZeDeviceMemAllocDescriptor;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.ZeDevicesHandle;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.ZeDriverHandle;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.ZeDriverProperties;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.ZeGroupDispatch;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.ZeHostMemAllocDescriptor;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.ZeInitFlag;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.ZeKernelDescriptor;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.ZeKernelHandle;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.ZeModuleDescriptor;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.ZeModuleFormat;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.ZeModuleHandle;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.ZeResult;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.Ze_Structure_Type;
import uk.ac.manchester.tornado.drivers.spirv.levelzero.utils.LevelZeroUtils;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.reflect.Array;
import java.util.Arrays;

/**
 * Kernel that runs:
 *
 * <code>
 *    __kernel void readWriteKernel(__global int* input_output) {
 *          uint idx = get_global_id(0);
 *          input_output[idx] = input_output[idx] + 5;
 *   }
 * </code>
 *
 *
 * To compile to SPIR-V:
 *
 * <code>
 *     $ clang -cc1 -triple spir readWriteKernel.cl -O0 -finclude-default-header -emit-llvm-bc -o readWriteKernel.bc
 *     $ llvm-spirv readWriteKernel.bc -o readWriteKernel.spv
 * </code>
 *
 *
 * How to run?
 *
 * <code>
 *     export LD_LIBRARY_PATH=/path/to/level-zero/build/lib:$LD_LIBRARY_PATH
 *     mvn clean compile package
 *     $JAVA_HOME/bin/java -Djava.library.path=/path/to/levelzero-jni/levelZeroLib/build/ -cp target/beehive-levelzero-jni-0.1.3.jar uk.ac.manchester.tornado.drivers.spirv.levelzero.benchmark.SharedMemBenchmark <mode> <numElements> <warmupIterations> <numIterations>
 *     <mode> = mmap | host | shared
 *     <numElements> = number of elements in the array. Elements are integers
 *     <warmupIterations> = number of warmup iterations
 *     <numIterations> = number of iterations
 * </code>
 */

public class SharedMemBenchmark {

    private static LevelZeroContext context;
    private static ZeDeviceMemAllocDescriptor deviceMemAllocDesc;
    private static LevelZeroDevice device;
    private static ZeHostMemAllocDescriptor hostMemAllocDesc;
    private static final String kernelFile = "readWriteKernel.spv";
    private static LevelZeroCommandList commandList;

    private static ZeCommandListHandle zeCommandListHandler;

    private static LevelZeroCommandList commandListIn;
    private static ZeCommandListHandle zeCommandListHandlerIn;
    private static LevelZeroCommandList commandListOut;
    private static ZeCommandListHandle zeCommandListHandlerOut;
    private static LevelZeroCommandQueue commandQueue;
    private static ZeCommandQueueHandle commandQueueHandle;

    private static void setup_levelzero() {
        LevelZeroDriver driver = new LevelZeroDriver();
        int result = driver.zeInit(ZeInitFlag.ZE_INIT_FLAG_GPU_ONLY);
        LevelZeroUtils.errorLog("zeInit", result);

        int[] numDrivers = new int[1];
        result = driver.zeDriverGet(numDrivers, null);
        LevelZeroUtils.errorLog("zeDriverGet", result);

        ZeDriverHandle driverHandler = new ZeDriverHandle(numDrivers[0]);

        result = driver.zeDriverGet(numDrivers, driverHandler);
        LevelZeroUtils.errorLog("zeDriverGet", result);

        // ============================================
        // Create the Context
        // ============================================
        // Create context Description
        ZeContextDescriptor contextDescription = new ZeContextDescriptor();
        // Create context object
        context = new LevelZeroContext(driverHandler, contextDescription);
        // Call native method for creating the context
        result = context.zeContextCreate(driverHandler.getZe_driver_handle_t_ptr()[0]);
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
        device = driver.getDevice(driverHandler, 0);

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

        commandQueueHandle = new ZeCommandQueueHandle();
        commandQueue = new LevelZeroCommandQueue(context, commandQueueHandle);
        ZeCommandQueueDescriptor commandQueueDescription = new ZeCommandQueueDescriptor();

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
        // Create a kernel command list
        // ============================================
        zeCommandListHandler = new ZeCommandListHandle();
        commandList = new LevelZeroCommandList(context, zeCommandListHandler);
        ZeCommandListDescriptor commandListDescription = new ZeCommandListDescriptor();
        commandListDescription.setCommandQueueGroupOrdinal(commandQueueDescription.getOrdinal());
        result = context.zeCommandListCreate(context.getContextHandle().getContextPtr()[0], device.getDeviceHandlerPtr(), commandListDescription, zeCommandListHandler);
        LevelZeroUtils.errorLog("zeCommandListCreate", result);

        // Create a input copy command list
        zeCommandListHandlerIn = new ZeCommandListHandle();
        commandListIn = new LevelZeroCommandList(context, zeCommandListHandlerIn);
        result = context.zeCommandListCreate(context.getContextHandle().getContextPtr()[0], device.getDeviceHandlerPtr(), commandListDescription, zeCommandListHandlerIn);
        LevelZeroUtils.errorLog("zeCommandListCreate", result);

        // Create a output copy command list
        zeCommandListHandlerOut = new ZeCommandListHandle();
        commandListOut = new LevelZeroCommandList(context, zeCommandListHandlerOut);
        result = context.zeCommandListCreate(context.getContextHandle().getContextPtr()[0], device.getDeviceHandlerPtr(), commandListDescription, zeCommandListHandlerOut);
        LevelZeroUtils.errorLog("zeCommandListCreate", result);

        deviceMemAllocDesc = new ZeDeviceMemAllocDescriptor();
        // deviceMemAllocDesc.setFlags(ZeDeviceMemAllocFlags.ZE_DEVICE_MEM_ALLOC_FLAG_BIAS_UNCACHED);
        deviceMemAllocDesc.setOrdinal(0);

        hostMemAllocDesc = new ZeHostMemAllocDescriptor();
        // hostMemAllocDesc.setFlags(ZeHostMemAllocFlags.ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED);
    }

    private static void setupReadWriteKernel(long argAddress, int num_elements) {
        ZeModuleHandle module = new ZeModuleHandle();
        ZeModuleDescriptor moduleDesc = new ZeModuleDescriptor();
        ZeBuildLogHandle buildLog = new ZeBuildLogHandle();
        moduleDesc.setFormat(ZeModuleFormat.ZE_MODULE_FORMAT_IL_SPIRV);
        moduleDesc.setBuildFlags("");

        int result = context.zeModuleCreate(context.getDefaultContextPtr(), device.getDeviceHandlerPtr(), moduleDesc, module, buildLog, kernelFile);
        LevelZeroUtils.errorLog("zeModuleCreate", result);

        if (result != ZeResult.ZE_RESULT_SUCCESS) {
            // Print Logs
            int[] sizeLog = new int[1];
            String[] errorMessage = new String[1];
            result = context.zeModuleBuildLogGetString(buildLog, sizeLog, errorMessage);
            System.out.println("LOGS::: " + sizeLog[0] + "  -- " + errorMessage[0]);
            LevelZeroUtils.errorLog("zeModuleBuildLogGetString", result);
        }


        // Create Module Object
        LevelZeroModule levelZeroModule = new LevelZeroModule(module, moduleDesc, buildLog);

        // Destroy Log
        result = levelZeroModule.zeModuleBuildLogDestroy(buildLog);
        LevelZeroUtils.errorLog("zeModuleBuildLogDestroy", result);

        ZeKernelDescriptor kernelDesc = new ZeKernelDescriptor();
        ZeKernelHandle kernel = new ZeKernelHandle();
        kernelDesc.setKernelName("readWriteKernel");
        result = levelZeroModule.zeKernelCreate(module.getPtrZeModuleHandle(), kernelDesc, kernel);
        LevelZeroUtils.errorLog("zeKernelCreate", result);

        // We create a kernel Object
        LevelZeroKernel levelZeroKernel = new LevelZeroKernel(kernelDesc, kernel, levelZeroModule);

        // Prepare kernel for launch
        // A) Suggest scheduling parameters to level-zero
        int[] groupSizeX = new int[] { 32 };
        int[] groupSizeY = new int[] { 1 };
        int[] groupSizeZ = new int[] { 1 };
        result = levelZeroKernel.zeKernelSuggestGroupSize(kernel.getPtrZeKernelHandle(), num_elements, 1, 1, groupSizeX, groupSizeY, groupSizeZ);
        LevelZeroUtils.errorLog("zeKernelSuggestGroupSize", result);

        result = levelZeroKernel.zeKernelSetGroupSize(kernel.getPtrZeKernelHandle(), groupSizeX, groupSizeY, groupSizeZ);
        LevelZeroUtils.errorLog("zeKernelSetGroupSize", result);

        result = levelZeroKernel.zeKernelSetArgumentValue(kernel.getPtrZeKernelHandle(), 0, Sizeof.POINTER.getNumBytes(), argAddress);
        LevelZeroUtils.errorLog("zeKernelSetArgumentValue", result);

        int groupCountX = num_elements / groupSizeX[0];
        System.out.println("groupSizeX = " + Arrays.toString(groupSizeX));
        System.out.println("groupCountX = " + groupCountX);
        System.out.println("num_elements = " + num_elements);
        if (num_elements % groupSizeX[0] != 0) {
            throw new RuntimeException("Number of elements must be a multiple of groupSizeX");
        }

        // Dispatch SPIR-V Kernel
        ZeGroupDispatch dispatch = new ZeGroupDispatch();
        dispatch.setGroupCountX((long) num_elements / groupSizeX[0]);
        dispatch.setGroupCountY(1);
        dispatch.setGroupCountZ(1);

        // Launch the kernel on the Intel Integrated GPU
        result = commandList.zeCommandListAppendLaunchKernel(zeCommandListHandler.getPtrZeCommandListHandle(), kernel.getPtrZeKernelHandle(), dispatch, null, 0, null);
        LevelZeroUtils.errorLog("zeCommandListAppendLaunchKernel", result);
    }

    private static void mmapMemBenchmark(int numIntElements, int warmupIterations, int iterations) {
    /*
        1. Allocate mmaped memory through an arena (MemorySegment)
        2. Allocate device memory
        3. Write to all the pages in the memory segment on the host. This step should map the pages on the host
        4. Transfer the memory to the device
        5. Run the kernel
        6. Transfer back the data from the device to the host
        7. Read from all the pages in the memory segment on the host
     */

        long bufferSize = (long) numIntElements * Integer.BYTES;
        System.out.println("Buffer size: " + bufferSize + " bytes");

        // Allocate memory segment on the host
        Arena arena = Arena.ofShared();
        MemorySegment bufferSegment = arena.allocate(bufferSize, 8);
        LevelZeroByteBuffer hostBuffer = new LevelZeroByteBuffer(bufferSegment.address(), bufferSize, 8);

        // Allocate device memory
        LevelZeroByteBuffer deviceBuffer = new LevelZeroByteBuffer();
        int result = context.zeMemAllocDevice(context.getContextHandle().getContextPtr()[0], deviceMemAllocDesc, bufferSize, 8, device.getDeviceHandlerPtr(), deviceBuffer);
        LevelZeroUtils.errorLog("zeMemAllocDevice", result);

        setupReadWriteKernel(deviceBuffer.getPtrBuffer(), numIntElements);

        // Close the cmd list for the kernel
        result = commandList.zeCommandListClose(zeCommandListHandler.getPtrZeCommandListHandle());
        LevelZeroUtils.errorLog("zeCommandListClose", result);

        // Append copy input data to device
        result = commandListIn.zeCommandListAppendMemoryCopy(zeCommandListHandlerIn.getPtrZeCommandListHandle(), deviceBuffer, hostBuffer, (int) bufferSize, null, 0, null);
        LevelZeroUtils.errorLog("zeCommandListAppendMemoryCopy", result);
        result = commandListIn.zeCommandListClose(zeCommandListHandlerIn.getPtrZeCommandListHandle());
        LevelZeroUtils.errorLog("zeCommandListClose", result);

        // Append copy output data to host
        result = commandListOut.zeCommandListAppendMemoryCopy(zeCommandListHandlerOut.getPtrZeCommandListHandle(), hostBuffer, deviceBuffer, (int) bufferSize, null, 0, null);
        LevelZeroUtils.errorLog("zeCommandListAppendMemoryCopy", result);
        result = commandListOut.zeCommandListClose(zeCommandListHandlerOut.getPtrZeCommandListHandle());
        LevelZeroUtils.errorLog("zeCommandListClose", result);

        long[] fill_time = new long[warmupIterations + iterations];
        long[] host_to_device = new long[warmupIterations + iterations];
        long [] kernel_time = new long[warmupIterations + iterations];
        long[] device_to_host = new long[warmupIterations + iterations];
        long [] read_time = new long[warmupIterations + iterations];

        for (int i = 0; i < warmupIterations + iterations; i++) {
            fill_time[i] = System.nanoTime();
            writeSegmentHost(bufferSegment, numIntElements);
            fill_time[i] = System.nanoTime() - fill_time[i];

            host_to_device[i] = System.nanoTime();
            result = commandQueue.zeCommandQueueExecuteCommandLists(commandQueueHandle.getCommandQueueHandlerPointer(), 1, zeCommandListHandlerIn, null);
            result = result | commandQueue.zeCommandQueueSynchronize(commandQueueHandle.getCommandQueueHandlerPointer(), Long.MAX_VALUE);
            host_to_device[i] = System.nanoTime() - host_to_device[i];
            LevelZeroUtils.errorLog("zeCommandQueueExecuteCommandLists", result);

            kernel_time[i] = System.nanoTime();
            result = commandQueue.zeCommandQueueExecuteCommandLists(commandQueueHandle.getCommandQueueHandlerPointer(), 1, zeCommandListHandler, null);
            result = result | commandQueue.zeCommandQueueSynchronize(commandQueueHandle.getCommandQueueHandlerPointer(), Long.MAX_VALUE);
            kernel_time[i] = System.nanoTime() - kernel_time[i];
            LevelZeroUtils.errorLog("zeCommandQueueExecuteCommandLists", result);

            device_to_host[i] = System.nanoTime();
            result = commandQueue.zeCommandQueueExecuteCommandLists(commandQueueHandle.getCommandQueueHandlerPointer(), 1, zeCommandListHandlerOut, null);
            result = result | commandQueue.zeCommandQueueSynchronize(commandQueueHandle.getCommandQueueHandlerPointer(), Long.MAX_VALUE);
            device_to_host[i] = System.nanoTime() - device_to_host[i];
            LevelZeroUtils.errorLog("zeCommandQueueExecuteCommandLists", result);

            read_time[i] = System.nanoTime();
            readSegmentHost(bufferSegment, numIntElements);
            read_time[i] = System.nanoTime() - read_time[i];
        }

        report(iterations, warmupIterations, fill_time, host_to_device, kernel_time, device_to_host, read_time);

        arena.close();
    }

    private static void hostMemBenchmark(int numIntElements, int warmupIterations, int iterations) {
            /*
        1. Allocate host memory through level zero
        2. Allocate device memory
        3. Write to all the pages in the memory segment on the host. This step should map the pages on the host
        4. Transfer the memory to the device
        5. Run the kernel
        6. Transfer back the data from the device to the host
        7. Read from all the pages in the memory segment on the host
     */

        long bufferSize = (long) numIntElements * Integer.BYTES;
        System.out.println("Buffer size: " + bufferSize + " bytes");

        // Allocate memory segment on the host
        LevelZeroByteBuffer hostBuffer = new LevelZeroByteBuffer((int) bufferSize, 8);
        int result = context.zeMemAllocHost(context.getContextHandle().getContextPtr()[0], hostMemAllocDesc, bufferSize, 8, hostBuffer);
        LevelZeroUtils.errorLog("zeMemAllocHost", result);
        MemorySegment bufferSegment = MemorySegment.ofAddress(hostBuffer.getPtrBuffer()).reinterpret(bufferSize);

        // Allocate device memory
        LevelZeroByteBuffer deviceBuffer = new LevelZeroByteBuffer();
        result = context.zeMemAllocDevice(context.getContextHandle().getContextPtr()[0], deviceMemAllocDesc, bufferSize, 8, device.getDeviceHandlerPtr(), deviceBuffer);
        LevelZeroUtils.errorLog("zeMemAllocDevice", result);

        setupReadWriteKernel(deviceBuffer.getPtrBuffer(), numIntElements);

        // Close the cmd list for the kernel
        result = commandList.zeCommandListClose(zeCommandListHandler.getPtrZeCommandListHandle());
        LevelZeroUtils.errorLog("zeCommandListClose", result);

        // Append copy input data to device
        result = commandListIn.zeCommandListAppendMemoryCopy(zeCommandListHandlerIn.getPtrZeCommandListHandle(), deviceBuffer, hostBuffer, (int) bufferSize, null, 0, null);
        LevelZeroUtils.errorLog("zeCommandListAppendMemoryCopy", result);
        result = commandListIn.zeCommandListClose(zeCommandListHandlerIn.getPtrZeCommandListHandle());
        LevelZeroUtils.errorLog("zeCommandListClose", result);

        // Append copy output data to host
        result = commandListOut.zeCommandListAppendMemoryCopy(zeCommandListHandlerOut.getPtrZeCommandListHandle(), hostBuffer, deviceBuffer, (int) bufferSize, null, 0, null);
        LevelZeroUtils.errorLog("zeCommandListAppendMemoryCopy", result);
        result = commandListOut.zeCommandListClose(zeCommandListHandlerOut.getPtrZeCommandListHandle());
        LevelZeroUtils.errorLog("zeCommandListClose", result);

        long[] fill_time = new long[warmupIterations + iterations];
        long[] host_to_device = new long[warmupIterations + iterations];
        long [] kernel_time = new long[warmupIterations + iterations];
        long[] device_to_host = new long[warmupIterations + iterations];
        long [] read_time = new long[warmupIterations + iterations];

        for (int i = 0; i < warmupIterations + iterations; i++) {
            fill_time[i] = System.nanoTime();
            writeSegmentHost(bufferSegment, numIntElements);
            fill_time[i] = System.nanoTime() - fill_time[i];

            host_to_device[i] = System.nanoTime();
            result = commandQueue.zeCommandQueueExecuteCommandLists(commandQueueHandle.getCommandQueueHandlerPointer(), 1, zeCommandListHandlerIn, null);
            result = result | commandQueue.zeCommandQueueSynchronize(commandQueueHandle.getCommandQueueHandlerPointer(), Long.MAX_VALUE);
            host_to_device[i] = System.nanoTime() - host_to_device[i];
            LevelZeroUtils.errorLog("zeCommandQueueExecuteCommandLists", result);

            kernel_time[i] = System.nanoTime();
            result = commandQueue.zeCommandQueueExecuteCommandLists(commandQueueHandle.getCommandQueueHandlerPointer(), 1, zeCommandListHandler, null);
            result = result | commandQueue.zeCommandQueueSynchronize(commandQueueHandle.getCommandQueueHandlerPointer(), Long.MAX_VALUE);
            kernel_time[i] = System.nanoTime() - kernel_time[i];
            LevelZeroUtils.errorLog("zeCommandQueueExecuteCommandLists", result);

            device_to_host[i] = System.nanoTime();
            result = commandQueue.zeCommandQueueExecuteCommandLists(commandQueueHandle.getCommandQueueHandlerPointer(), 1, zeCommandListHandlerOut, null);
            result = result | commandQueue.zeCommandQueueSynchronize(commandQueueHandle.getCommandQueueHandlerPointer(), Long.MAX_VALUE);
            device_to_host[i] = System.nanoTime() - device_to_host[i];
            LevelZeroUtils.errorLog("zeCommandQueueExecuteCommandLists", result);

            read_time[i] = System.nanoTime();
            readSegmentHost(bufferSegment, numIntElements);
            read_time[i] = System.nanoTime() - read_time[i];
        }

        report(iterations, warmupIterations, fill_time, host_to_device, kernel_time, device_to_host, read_time);
    }

    private static void sharedMemBenchmark(int numIntElements, int warmupIterations, int iterations) {
        /*
        1. Allocate shared memory
        2. Write to all the pages in the shared memory on the host. This step should map the shared memory on the host
        3. Read + Write to all pages in the shared memory on the device (by adding a constant to each element).
        This step should transfer the shared memory to the device and invalidate the host mapping (due to write)
        4. Read from all the pages in the shared memory on the host.
        This step should transfer the shared memory back to the host

        This should be the worst case scenario for shared memory.
         */

        long bufferSize = (long) numIntElements * Integer.BYTES;
        System.out.println("Buffer size: " + bufferSize + " bytes");

        // Allocate shared memory

        LevelZeroBufferInteger buffer = new LevelZeroBufferInteger();
        int result = context.zeMemAllocShared(context.getContextHandle().getContextPtr()[0], deviceMemAllocDesc, hostMemAllocDesc, bufferSize, 8, device.getDeviceHandlerPtr(), buffer);
        LevelZeroUtils.errorLog("zeMemAllocShared", result);

        MemorySegment bufferSegment = MemorySegment.ofAddress(buffer.getPtrBuffer()).reinterpret(bufferSize);
        setupReadWriteKernel(bufferSegment.address(), numIntElements);

        // Close the cmd list
        result = commandList.zeCommandListClose(zeCommandListHandler.getPtrZeCommandListHandle());
        LevelZeroUtils.errorLog("zeCommandListClose", result);

        long[] fill_time = new long[warmupIterations + iterations];
        long[] host_to_device = new long[warmupIterations + iterations];
        long [] kernel_time = new long[warmupIterations + iterations];
        long[] device_to_host = new long[warmupIterations + iterations];
        long [] read_time = new long[warmupIterations + iterations];

        for (int i = 0; i < warmupIterations + iterations; i++) {
            fill_time[i] = System.nanoTime();
            writeSegmentHost(bufferSegment, numIntElements);
            fill_time[i] = System.nanoTime() - fill_time[i];

            kernel_time[i] = System.nanoTime();
            result = commandQueue.zeCommandQueueExecuteCommandLists(commandQueueHandle.getCommandQueueHandlerPointer(), 1, zeCommandListHandler, null);
            result = result | commandQueue.zeCommandQueueSynchronize(commandQueueHandle.getCommandQueueHandlerPointer(), Long.MAX_VALUE);
            kernel_time[i] = System.nanoTime() - kernel_time[i];
            LevelZeroUtils.errorLog("zeCommandQueueExecuteCommandLists", result);

            read_time[i] = System.nanoTime();
            readSegmentHost(bufferSegment, numIntElements);
            read_time[i] = System.nanoTime() - read_time[i];
        }

        report(iterations, warmupIterations, fill_time, host_to_device, kernel_time, device_to_host, read_time);
    }

    private static void writeSegmentHost(MemorySegment segment, int numIntElements) {
        for (int i = 0; i < numIntElements; i++) {
            long offset = i * ValueLayout.JAVA_INT.byteSize();
            segment.set(ValueLayout.JAVA_INT, offset, 10);
        }
    }

    private static void readSegmentHost(MemorySegment segment, int numIntElements) {
        for (int i = 0; i < numIntElements; i++) {
            long offset = i * ValueLayout.JAVA_INT.byteSize();
            int value = segment.get(ValueLayout.JAVA_INT, offset);
            if (value != 10 + 5) {
                System.out.println("Invalid value at index: " + i + " value: " + value);
            }
        }
    }

    private static void report(int num_iterations, int warmupIterations, long[] fillTime, long[] host_to_device,
                               long[] kernelTime, long[] device_to_host, long[] readTime) {
        long averageFill = 0;
        long deviceToHost = 0;
        long averageKernel = 0;
        long hostToDevice = 0;
        long averageRead = 0;
        for (int i = warmupIterations; i < warmupIterations + num_iterations; i++) {
            averageFill += fillTime[i];
            deviceToHost += device_to_host[i];
            averageKernel += kernelTime[i];
            hostToDevice += host_to_device[i];
            averageRead += readTime[i];
        }
        averageFill /= num_iterations;
        deviceToHost /= num_iterations;
        averageKernel /= num_iterations;
        hostToDevice /= num_iterations;
        averageRead /= num_iterations;
        long averageTotal = averageFill + deviceToHost + averageKernel + hostToDevice + averageRead;
        System.out.println("Average fill time: " + averageFill + " ns");
        System.out.println("Average device to host time: " + deviceToHost + " ns");
        System.out.println("Average kernel time: " + averageKernel + " ns");
        System.out.println("Average host to device time: " + hostToDevice + " ns");
        System.out.println("Average read time: " + averageRead + " ns");
        System.out.println("Average total time: " + averageTotal + " ns");
    }

    public static void main(String[] args) {
        if (args.length != 4) {
            System.out.println("Usage: SharedMemBenchmark <mode> <numElements> <warmupIterations> <numIterations>");
            System.exit(1);
        }
        String mode = args[0];
        int numIntElements = Integer.parseInt(args[1]);
        int warmupIterations = Integer.parseInt(args[2]);
        int numIterations = Integer.parseInt(args[3]);
        System.out.println("Running SharedMemBenchmark with mode: " + mode + " numElements: " + numIntElements +
                " warmupIterations: " + warmupIterations + " numIterations: " + numIterations);
        System.out.println("Data size: " + numIntElements * Integer.BYTES + " bytes");

        assert (int) ((long) numIntElements * Integer.BYTES) == numIntElements * Integer.BYTES : "Buffer size exceeds 2^31-1 bytes";

        setup_levelzero();

        switch (mode) {
            case "shared":
                sharedMemBenchmark(numIntElements, warmupIterations, numIterations);
                break;
            case "host":
                hostMemBenchmark(numIntElements, warmupIterations, numIterations);
                break;
            case "mmap":
                mmapMemBenchmark(numIntElements, warmupIterations, numIterations);
                break;
            default:
                System.out.println("Invalid mode: " + mode);
                System.exit(1);
        }
    }
}
