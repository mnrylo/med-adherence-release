plugins {
    `java-library`
    kotlin("jvm")
}

java {
    sourceCompatibility = JavaVersion.VERSION_11
    targetCompatibility = JavaVersion.VERSION_11
}

kotlin {
    compilerOptions { jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_11) }
}

val ortVersion = "1.23.2"

dependencies {
    compileOnly("org.json:json:20250517")
    compileOnly("com.microsoft.onnxruntime:onnxruntime:$ortVersion")
    testImplementation("junit:junit:4.13.2")
    testImplementation("org.json:json:20250517")
    testImplementation("com.microsoft.onnxruntime:onnxruntime:$ortVersion")
}

tasks.test {
    systemProperty("bundle.dir", rootProject.file("bundle").absolutePath)
    if (project.hasProperty("replay")) {
        systemProperty("replay.dir", rootProject.file("../outputs/embedded/phase1/replay").absolutePath)
        systemProperty("reference.dir", rootProject.file("../outputs/embedded/phase1/reference").absolutePath)
        outputs.upToDateWhen { false }
    }
    maxHeapSize = "2g"
    testLogging { events("passed", "failed"); showStandardStreams = true }
}
