import { Routes } from '@angular/router';
import { HomeComponent } from './Components/home/home.component';
import { GalleryComponent } from './Components/gallery/gallery.component';
import { EditorComponent } from './Components/editor/editor.component';

export const routes: Routes = [
  { path: '', redirectTo: 'home', pathMatch: 'full' },
  { path: 'home', component: HomeComponent },
  { path: 'gallery', component: GalleryComponent },
  { path: 'editor', component: EditorComponent },
  { path: '**', redirectTo: 'home' },
];
