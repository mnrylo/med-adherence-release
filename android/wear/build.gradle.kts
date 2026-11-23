plugins {
    id("com.android.application")
    kotlin("android")
}

android {
    namespace = "com.medmon.emb.wear"
    compileSdk = 36
    defaultConfig {
        applicationId = "com.medmon.embbench"
        minSdk = 30
        targetSdk = 34
        versionCode = 1
        versionName = "0.1"
        ndk { abiFilters += listOf("arm64-v8a", "armeabi-v7a") + if (project.hasProperty("emulator")) listOf("x86_64") else listOf() }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    packaging { jniLibs { useLegacyPackaging = false } }
}

kotlin {
    compilerOptions { jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_11) }
}

dependencies {
    implementation(project(":benchlib"))
}
