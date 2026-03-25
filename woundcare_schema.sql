-- AI-Powered Surgical Wound Care Tool FULL Database Schema
-- Includes custom app (api) and Django internal tables (auth, contenttypes)
-- Generated from Django Migrations

-- =============================================================
-- APP: api
-- =============================================================

-- Migration api.0001
--
-- Create model User
--
CREATE TABLE `users` (`id` bigint AUTO_INCREMENT NOT NULL PRIMARY KEY, `password` varchar(128) NOT NULL, `last_login` datetime(6) NULL, `name` varchar(100) NOT NULL, `email` varchar(100) NOT NULL UNIQUE, `phone` varchar(20) NULL, `date_of_birth` varchar(50) NULL, `blood_type` varchar(10) NULL, `emergency_contact` varchar(100) NULL, `emergency_phone` varchar(20) NULL, `profile_image` varchar(255) NULL, `email_verified` bool NOT NULL, `created_at` datetime(6) NOT NULL);
--
-- Create model Classification
--
CREATE TABLE `classifications` (`id` bigint AUTO_INCREMENT NOT NULL PRIMARY KEY, `wound_type` varchar(100) NOT NULL, `confidence` double precision NOT NULL, `all_probabilities` json NOT NULL, `processing_time_ms` integer NULL, `timestamp` datetime(6) NOT NULL);
--
-- Create model EmailVerificationOTP
--
CREATE TABLE `email_verification_otps` (`id` bigint AUTO_INCREMENT NOT NULL PRIMARY KEY, `email` varchar(100) NOT NULL, `otp_code` varchar(6) NOT NULL, `expires_at` datetime(6) NOT NULL, `verified` bool NOT NULL, `created_at` datetime(6) NOT NULL, `pending_name` varchar(100) NULL, `pending_password_hash` varchar(255) NULL);
--
-- Create model Case
--
CREATE TABLE `cases` (`id` bigint AUTO_INCREMENT NOT NULL PRIMARY KEY, `name` varchar(255) NOT NULL, `description` longtext NULL, `created_at` datetime(6) NOT NULL, `user_id` bigint NOT NULL);
--
-- Create model Recommendation
--
CREATE TABLE `recommendations` (`id` bigint AUTO_INCREMENT NOT NULL PRIMARY KEY, `summary` longtext NOT NULL, `cleaning_instructions` json NULL, `dressing_recommendations` json NULL, `medication_suggestions` json NULL, `expected_healing_time` varchar(100) NULL, `follow_up_schedule` json NULL, `warning_signs` json NULL, `when_to_seek_help` json NULL, `diet_advice` json NULL, `activity_restrictions` json NULL, `ai_confidence` integer NULL, `created_at` datetime(6) NOT NULL, `classification_id` bigint NOT NULL);
--
-- Create model UserSession
--
CREATE TABLE `sessions` (`id` bigint AUTO_INCREMENT NOT NULL PRIMARY KEY, `session_token` varchar(255) NOT NULL UNIQUE, `expires_at` datetime(6) NOT NULL, `created_at` datetime(6) NOT NULL, `user_id` bigint NOT NULL);
--
-- Create model Wound
--
CREATE TABLE `wounds` (`id` bigint AUTO_INCREMENT NOT NULL PRIMARY KEY, `image_path` varchar(255) NOT NULL, `original_filename` varchar(255) NULL, `upload_date` datetime(6) NOT NULL, `status` varchar(50) NOT NULL, `notes` longtext NULL, `classification` varchar(100) NULL, `confidence` double precision NULL, `redness_level` integer NULL, `discharge_detected` bool NULL, `discharge_type` varchar(50) NULL, `edge_quality` integer NULL, `tissue_composition` json NULL, `analysis` json NULL, `case_id` bigint NULL, `user_id` bigint NOT NULL);
--
-- Create model Comparison
--
CREATE TABLE `comparisons` (`id` bigint AUTO_INCREMENT NOT NULL PRIMARY KEY, `analysis` json NULL, `created_at` datetime(6) NOT NULL, `case_id` bigint NULL, `wound_after_id` bigint NULL, `wound_before_id` bigint NULL);
--
-- Add field wound to classification
--
ALTER TABLE `classifications` ADD COLUMN `wound_id` bigint NOT NULL , ADD CONSTRAINT `classifications_wound_id_fc3d1c67_fk_wounds_id` FOREIGN KEY (`wound_id`) REFERENCES `wounds`(`id`);
CREATE INDEX `classifications_wound_type_af873151` ON `classifications` (`wound_type`);
CREATE INDEX `email_verification_otps_email_f4e8230f` ON `email_verification_otps` (`email`);
ALTER TABLE `cases` ADD CONSTRAINT `cases_user_id_8b67f79f_fk_users_id` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`);
ALTER TABLE `recommendations` ADD CONSTRAINT `recommendations_classification_id_f676690f_fk_classifications_id` FOREIGN KEY (`classification_id`) REFERENCES `classifications` (`id`);
ALTER TABLE `sessions` ADD CONSTRAINT `sessions_user_id_05e26f4a_fk_users_id` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`);
ALTER TABLE `wounds` ADD CONSTRAINT `wounds_case_id_57ec7b86_fk_cases_id` FOREIGN KEY (`case_id`) REFERENCES `cases` (`id`);
ALTER TABLE `wounds` ADD CONSTRAINT `wounds_user_id_711f1bbc_fk_users_id` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`);
CREATE INDEX `wounds_upload_date_1869410e` ON `wounds` (`upload_date`);
ALTER TABLE `comparisons` ADD CONSTRAINT `comparisons_case_id_9101e9b7_fk_cases_id` FOREIGN KEY (`case_id`) REFERENCES `cases` (`id`);
ALTER TABLE `comparisons` ADD CONSTRAINT `comparisons_wound_after_id_b43fbc4d_fk_wounds_id` FOREIGN KEY (`wound_after_id`) REFERENCES `wounds` (`id`);
ALTER TABLE `comparisons` ADD CONSTRAINT `comparisons_wound_before_id_72719921_fk_wounds_id` FOREIGN KEY (`wound_before_id`) REFERENCES `wounds` (`id`);


-- Migration api.0002
--
-- Add field gender to user
--
ALTER TABLE `users` ADD COLUMN `gender` varchar(20) NULL;


-- Migration api.0003
--
-- Add field age to user
--
ALTER TABLE `users` ADD COLUMN `age` integer NULL;


-- Migration api.0004
--
-- Add field pending_age to emailverificationotp
--
ALTER TABLE `email_verification_otps` ADD COLUMN `pending_age` integer NULL;
--
-- Add field pending_gender to emailverificationotp
--
ALTER TABLE `email_verification_otps` ADD COLUMN `pending_gender` varchar(20) NULL;
--
-- Add field pending_phone to emailverificationotp
--
ALTER TABLE `email_verification_otps` ADD COLUMN `pending_phone` varchar(20) NULL;


-- Migration api.0005
--
-- Add field status to case
--
ALTER TABLE `cases` ADD COLUMN `status` varchar(50) DEFAULT 'active' NOT NULL;
ALTER TABLE `cases` ALTER COLUMN `status` DROP DEFAULT;


-- Migration api.0006
--
-- Add field is_confirmed to wound
--
ALTER TABLE `wounds` ADD COLUMN `is_confirmed` bool DEFAULT b'0' NOT NULL;
ALTER TABLE `wounds` ALTER COLUMN `is_confirmed` DROP DEFAULT;
CREATE INDEX `wounds_is_confirmed_10c7ca33` ON `wounds` (`is_confirmed`);


-- =============================================================
-- APP: auth
-- =============================================================

-- Migration auth.0001
--
-- Create model Permission
--
CREATE TABLE `auth_permission` (`id` integer AUTO_INCREMENT NOT NULL PRIMARY KEY, `name` varchar(50) NOT NULL, `content_type_id` integer NOT NULL, `codename` varchar(100) NOT NULL);
--
-- Create model Group
--
CREATE TABLE `auth_group` (`id` integer AUTO_INCREMENT NOT NULL PRIMARY KEY, `name` varchar(80) NOT NULL UNIQUE);
CREATE TABLE `auth_group_permissions` (`id` bigint AUTO_INCREMENT NOT NULL PRIMARY KEY, `group_id` integer NOT NULL, `permission_id` integer NOT NULL);
--
-- Create model User
--
-- (no-op)
ALTER TABLE `auth_permission` ADD CONSTRAINT `auth_permission_content_type_id_codename_01ab375a_uniq` UNIQUE (`content_type_id`, `codename`);
ALTER TABLE `auth_permission` ADD CONSTRAINT `auth_permission_content_type_id_2f476e4b_fk_django_co` FOREIGN KEY (`content_type_id`) REFERENCES `django_content_type` (`id`);
ALTER TABLE `auth_group_permissions` ADD CONSTRAINT `auth_group_permissions_group_id_permission_id_0cd325b0_uniq` UNIQUE (`group_id`, `permission_id`);
ALTER TABLE `auth_group_permissions` ADD CONSTRAINT `auth_group_permissions_group_id_b120cbf9_fk_auth_group_id` FOREIGN KEY (`group_id`) REFERENCES `auth_group` (`id`);
ALTER TABLE `auth_group_permissions` ADD CONSTRAINT `auth_group_permissio_permission_id_84c5c92e_fk_auth_perm` FOREIGN KEY (`permission_id`) REFERENCES `auth_permission` (`id`);


-- Migration auth.0002
--
-- Alter field name on permission
--
ALTER TABLE `auth_permission` MODIFY `name` varchar(255) NOT NULL;


-- Migration auth.0003
--
-- Alter field email on user
--
-- (no-op)


-- Migration auth.0004
--
-- Alter field username on user
--
-- (no-op)


-- Migration auth.0005
--
-- Alter field last_login on user
--
-- (no-op)


-- Migration auth.0006


-- Migration auth.0007
--
-- Alter field username on user
--
-- (no-op)


-- Migration auth.0008
--
-- Alter field username on user
--
-- (no-op)


-- Migration auth.0009
--
-- Alter field last_name on user
--
-- (no-op)


-- Migration auth.0010
--
-- Alter field name on group
--
ALTER TABLE `auth_group` MODIFY `name` varchar(150) NOT NULL;


-- Migration auth.0011
--
-- Raw Python operation
--
-- THIS OPERATION CANNOT BE WRITTEN AS SQL


-- Migration auth.0012
--
-- Alter field first_name on user
--
-- (no-op)


-- =============================================================
-- APP: contenttypes
-- =============================================================

-- Migration contenttypes.0001
--
-- Create model ContentType
--
CREATE TABLE `django_content_type` (`id` integer AUTO_INCREMENT NOT NULL PRIMARY KEY, `name` varchar(100) NOT NULL, `app_label` varchar(100) NOT NULL, `model` varchar(100) NOT NULL);
--
-- Alter unique_together for contenttype (1 constraint(s))
--
ALTER TABLE `django_content_type` ADD CONSTRAINT `django_content_type_app_label_model_76bd3d3b_uniq` UNIQUE (`app_label`, `model`);


-- Migration contenttypes.0002
--
-- Change Meta options on contenttype
--
-- (no-op)
--
-- Alter field name on contenttype
--
ALTER TABLE `django_content_type` MODIFY `name` varchar(100) NULL;
--
-- Raw Python operation
--
-- THIS OPERATION CANNOT BE WRITTEN AS SQL
--
-- Remove field name from contenttype
--
ALTER TABLE `django_content_type` DROP COLUMN `name`;


