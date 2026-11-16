// AGP 8.9 is the first line that supports compileSdk 36 (Android 16, from 8.9.1); 8.9.3 is its last patch, with R8
// fixes. An app that builds this project through includeBuild must use the same AGP and Kotlin versions.
plugins {
    id("com.android.library") version "8.9.3" apply false
    id("com.android.application") version "8.9.3" apply false
    id("org.jetbrains.kotlin.android") version "2.0.21" apply false
}

// Optional: CZECHSPEECH_BUILD_ROOT moves every build directory elsewhere, e.g. off /mnt/c when building in WSL.
System.getenv("CZECHSPEECH_BUILD_ROOT")?.takeIf { it.isNotBlank() }?.let { buildRoot ->
    allprojects {
        layout.buildDirectory.set(file("$buildRoot/${project.name}"))
    }
}
