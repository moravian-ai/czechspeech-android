import org.jetbrains.kotlin.gradle.dsl.JvmTarget
import java.io.IOException
import java.util.concurrent.CompletableFuture
import java.util.concurrent.TimeUnit

plugins {
    id("com.android.library")
    id("org.jetbrains.kotlin.android")
    `maven-publish`
    signing
}

group = "ai.moravian"
version = "0.3.0"

val nativeOut = rootProject.layout.projectDirectory.dir("native/out")
val hostLibDir = nativeOut.dir("host/linux-x86_64")

android {
    namespace = "ai.moravian.czechspeech"
    compileSdk = 36
    // AGP 8.9 would use 35.0.0 by default; 36.0.0 goes with compileSdk 36.
    buildToolsVersion = "36.0.0"

    defaultConfig {
        minSdk = 26
        consumerProguardFiles("consumer-rules.pro")
    }

    // Prebuilt by native/build-native.sh (one whole library per CPU variant, chosen at runtime).
    sourceSets["main"].jniLibs.srcDir(nativeOut.dir("jniLibs"))

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }

    testOptions {
        unitTests.all { test ->
            // Host build of the JNI library for JVM tests; integration tests skip when it is missing.
            test.systemProperty("java.library.path", hostLibDir.asFile.absolutePath)
            test.maxHeapSize = "3g"
            test.maxParallelForks = 1
            test.testLogging {
                events("passed", "skipped", "failed")
                showStandardStreams = true
                exceptionFormat = org.gradle.api.tasks.testing.logging.TestExceptionFormat.FULL
            }
        }
    }

    publishing {
        singleVariant("release") {
            withSourcesJar()
            withJavadocJar()
        }
    }
}

kotlin {
    explicitApi()
    compilerOptions {
        jvmTarget.set(JvmTarget.JVM_11)
    }
}

dependencies {
    // The lowest version the SDK needs, so it is the lowest an app must have; an app resolves the newer one its other
    // libraries ask for (AndroidX core 1.18 brings 1.9.0). Lint's GradleDependency would raise everyone's floor.
    //noinspection GradleDependency
    api("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.8.1")

    testImplementation("junit:junit:4.13.2")
}

// LICENSE and THIRD_PARTY_NOTICES travel inside the AAR (META-INF/czechspeech-android/), next to the compiled
// whisper.cpp/ggml code and the Silero VAD model they cover.
abstract class CopyNotices : DefaultTask() {
    @get:InputFiles
    abstract val notices: ConfigurableFileCollection

    @get:OutputDirectory
    abstract val outputDir: DirectoryProperty

    @TaskAction
    fun copy() {
        val dir = outputDir.get().asFile.resolve("META-INF/czechspeech-android")
        dir.deleteRecursively()
        dir.mkdirs()
        notices.forEach { it.copyTo(dir.resolve(it.name), overwrite = true) }
    }
}

val copyNotices = tasks.register<CopyNotices>("copyNotices") {
    notices.from(rootProject.file("LICENSE"), rootProject.file("THIRD_PARTY_NOTICES"))
    outputDir.set(layout.buildDirectory.dir("generated/notices"))
}

androidComponents {
    onVariants { variant ->
        variant.sources.resources?.addGeneratedSourceDirectory(copyNotices, CopyNotices::outputDir)
    }
}

val expectedNativeLibs = listOf(
    "arm64-v8a/libczechspeech_armv8a.so",
    "arm64-v8a/libczechspeech_armv82_dotprod_fp16.so",
    "arm64-v8a/libczechspeech_armv82_dotprod_fp16_i8mm.so",
    "x86_64/libczechspeech_x86_64.so",
)

val checkNativeLibs by tasks.registering {
    description = "Fails when the prebuilt native libraries are missing or were built from other sources than the ones in " +
        "this tree (run native/build-native.sh android). Only the tasks that package the AAR depend on it, not the unit tests."
    val dir = nativeOut.dir("jniLibs").asFile
    val stampFile = nativeOut.file("source-stamp").asFile
    doLast {
        val missing = expectedNativeLibs.filterNot { dir.resolve(it).isFile }
        if (missing.isNotEmpty()) {
            throw GradleException("Missing native libraries in $dir: $missing. Run native/build-native.sh android first.")
        }
        // The libraries must be built from the native sources of this tree (a library without a newer native function throws
        // UnsatisfiedLinkError on the first call): native/build-native.sh writes the hash of them next to the libraries.
        // bash runs in the repository root with relative arguments with forward slashes: on Windows a bash does not open
        // C:\... paths (WSL's) or the script needs the tools of its own environment (Git's), and the script resolves
        // its paths from where it runs
        val rel = rootProject.projectDir.toPath().relativize(stampFile.toPath()).toString().replace(File.separatorChar, '/')
        val failure = "Cannot check the stamp of the native libraries. Run native/build-native.sh android (needs bash and sha256sum; on Windows use WSL or Git Bash)"
        val check = try {
            val process = ProcessBuilder("bash", "native/source-stamp.sh", "check", rel)
                .directory(rootProject.projectDir).redirectErrorStream(true).start()
            process.outputStream.close()
            val reader = CompletableFuture.supplyAsync { process.inputStream.bufferedReader().readText().trim() }
            if (!process.waitFor(60, TimeUnit.SECONDS)) {
                process.destroyForcibly()
                throw GradleException("$failure: the stamp check did not finish in 60 s.")
            }
            process.exitValue() to reader.get(5, TimeUnit.SECONDS)
        } catch (e: IOException) {
            throw GradleException("$failure: bash could not be started (${e.message}).")
        }
        if (check.first != 0) {
            throw GradleException(check.second.ifEmpty { "$failure: the script exited with ${check.first} and no output." })
        }
    }
}

tasks.matching { it.name == "bundleReleaseAar" || it.name == "bundleDebugAar" }.configureEach {
    dependsOn(checkNativeLibs)
}

// Static Maven repository in a folder (build/maven-repo, or -Pczechspeech.mavenRepoDir): ./gradlew
// :czechspeech:publishToLocalFolder. scripts/release-maven-central.sh signs it and zips it for Maven Central.
val mavenRepoDir = providers.gradleProperty("czechspeech.mavenRepoDir")
    .map { file(it) }
    .orElse(rootProject.layout.projectDirectory.dir("build/maven-repo").asFile)

publishing {
    publications {
        register<MavenPublication>("release") {
            groupId = "ai.moravian"
            artifactId = "czechspeech-android"
            version = project.version.toString()
            afterEvaluate { from(components["release"]) }
            pom {
                name.set("CzechSpeech for Android")
                description.set(
                    "Offline Czech speech-to-text for Android: the CzechSpeech model (fine-tuned Whisper " +
                        "large-v3-turbo, 4-bit) on whisper.cpp with Silero VAD. Audio never leaves the device."
                )
                url.set("https://moravian.ai/czechspeech/")
                inceptionYear.set("2026")
                licenses {
                    license {
                        name.set("PolyForm Noncommercial License 1.0.0")
                        url.set("https://polyformproject.org/licenses/noncommercial/1.0.0")
                        distribution.set("repo")
                        comments.set(
                            "Dual licence: free for noncommercial use (PolyForm Noncommercial 1.0.0); commercial licence: " +
                                "info@moravian.ai. Copyright (c) 2026 Jan Rosecký (Moravian AI)."
                        )
                    }
                    license {
                        name.set("CzechSpeech commercial licence")
                        url.set("https://moravian.ai/czechspeech/")
                        distribution.set("repo")
                        comments.set("Commercial use needs a separate licence from Moravian AI: info@moravian.ai.")
                    }
                }
                organization {
                    name.set("Moravian AI")
                    url.set("https://moravian.ai")
                }
                scm {
                    url.set("https://github.com/moravian-ai/czechspeech-android")
                    connection.set("scm:git:https://github.com/moravian-ai/czechspeech-android.git")
                    developerConnection.set("scm:git:https://github.com/moravian-ai/czechspeech-android.git")
                }
                developers {
                    developer {
                        id.set("moravianai")
                        name.set("Moravian AI")
                        email.set("info@moravian.ai")
                        organization.set("Moravian AI")
                        organizationUrl.set("https://moravian.ai")
                    }
                }
            }
        }
    }
    repositories {
        maven {
            name = "localFolder"
            url = uri(mavenRepoDir)
        }
    }
}

tasks.register("publishToLocalFolder") {
    group = "publishing"
    description = "Writes the release AAR, POM, sources jar, Gradle module metadata and checksums as a static Maven repository."
    dependsOn("publishReleasePublicationToLocalFolderRepository")
    doLast { println("Maven repository: ${mavenRepoDir.get()}") }
}

// Signing for Maven Central (scripts/release-maven-central.sh): off unless -Pczechspeech.sign=true, so normal
// builds, the tests and the sample app need no gpg. The key comes from -Psigning.gnupg.keyName=<key id>.
if (providers.gradleProperty("czechspeech.sign").orNull == "true") {
    signing {
        useGpgCmd()
        sign(publishing.publications["release"])
    }
}
