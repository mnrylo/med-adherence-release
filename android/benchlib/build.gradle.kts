plugins {
    id("com.android.library")
    kotlin("android")
}

android {
    namespace = "com.medmon.emb.bench"
    compileSdk = 36
    defaultConfig { minSdk = 30 }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    sourceSets["main"].assets.srcDir(layout.buildDirectory.dir("generated/bundleAssets"))
}

kotlin {
    compilerOptions { jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_11) }
}

val syncBundle by tasks.registering(Sync::class) {
    from(rootProject.file("bundle"))
    into(layout.buildDirectory.dir("generated/bundleAssets/bundle"))
}
tasks.named("preBuild") { dependsOn(syncBundle) }

dependencies {
    api(project(":core"))
    api("com.microsoft.onnxruntime:onnxruntime-android:1.23.2")
    api("com.google.android.gms:play-services-wearable:19.0.0")
}

dependencies {
    testImplementation("junit:junit:4.13.2")
    testImplementation("org.json:json:20250517")
}

android.testOptions.unitTests.all {
    it.systemProperty("bundle.dir", rootProject.file("bundle").absolutePath)
}
